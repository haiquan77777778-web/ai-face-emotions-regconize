"""
pipeline.py – ghép detect + crop + predict thành process_frame

Giao tiếp (đúng hợp đồng):
    process_frame(frame_bgr, detector="mediapipe", model=None)
        -> [{"box": (x, y, w, h), "label": str, "prob": float, "probs": dict}, ...]
     (detector "mediapipe" có thêm khóa "blendshapes": dict 52 giá trị của Face Landmarker)

  - Căn chỉnh mặt: detector mediapipe cho landmark -> xoay mặt cho hai mắt nằm ngang trước khi phân loại
    (Pipeline(align=False) hoặc phím a để tắt)
  - Làm mượt: trung bình xác suất SMOOTH_N khung gần nhất, lịch sử riêng cho từng mặt (ghép theo vị trí gần nhất)
  - Tối ưu: phát hiện mặt cách DETECT_EVERY khung, thu khung về rộng DETECT_WIDTH rồi quy đổi tọa độ, giới hạn MAX_FACES

Chạy webcam:  python pipeline.py      (a = bật/tắt căn chỉnh, q = thoát)
"""
import collections
import sys
import threading
import time

import cv2
import numpy as np

from detectors import detect_faces, crop_face, align_crop
from predict import predict, LABELS, DEFAULT_MODEL
from image_quality import analyze_face_quality, low_confidence_warning

DETECT_EVERY = 3
DETECT_WIDTH = 480
SMOOTH_N = 3
MAX_FACES = 5
MATCH_DIST = 80


class Pipeline:
    def __init__(self, smooth_n=SMOOTH_N, detect_every=DETECT_EVERY, align=True,
                 crop_margin=0.15, check_quality=True):
        if smooth_n < 1 or detect_every < 1 or not 0 <= crop_margin <= 0.5:
            raise ValueError("smooth_n/detect_every phải >= 1; crop_margin trong [0, 0.5].")
        self.smooth_n, self.detect_every, self.align = smooth_n, detect_every, align
        self.crop_margin = crop_margin
        self.check_quality = check_quality
        self.reset()
        self._ctx = None

    def reset(self):
        self._n = 0
        self._faces = []
        self._hist = {}          # key = tâm mặt (cx, cy) -> deque xác suất

    def _match(self, box, taken):
        x, y, w, h = box
        c = (x + w // 2, y + h // 2)
        best, bd = None, MATCH_DIST
        for k in self._hist:
            d = ((c[0] - k[0]) ** 2 + (c[1] - k[1]) ** 2) ** 0.5
            if k not in taken and d < bd:
                best, bd = k, d
        dq = self._hist.pop(best) if best is not None else collections.deque(maxlen=self.smooth_n)
        self._hist[c] = dq
        return c

    def process_frame(self, frame_bgr, detector="mediapipe", model=None):
        model = model or DEFAULT_MODEL
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        context = (detector, model, self.align, self.crop_margin, self.smooth_n, self.check_quality,
                   self.detect_every, frame_bgr.shape[:2])
        if self._ctx != context:                # đổi cách crop/model -> bỏ lịch sử cũ
            self.reset()
            self._ctx = context
        self._n += 1
        H, W = frame_bgr.shape[:2]

        if (self._n - 1) % self.detect_every == 0 or not self._faces:
            s = DETECT_WIDTH / W if W > DETECT_WIDTH else 1.0
            small = cv2.resize(frame_bgr, (int(W * s), int(H * s))) if s < 1 else frame_bgr
            self._faces = [dict(f, box=tuple(int(v / s) for v in f["box"]))
                           for f in detect_faces(small, detector)]

        faces = self._faces[:MAX_FACES]
        if not faces:
            self._hist.clear()
            return []

        out, taken = [], set()
        for f in faces:
            box = f["box"]
            face = self.crop_for_face(frame_bgr, f)
            if face.shape[0] < 20 or face.shape[1] < 20:
                continue
            _, _, probs = predict(face, model)
            key = self._match(box, taken)
            taken.add(key)
            self._hist[key].append(probs)
            avg = {l: float(np.mean([d[l] for d in self._hist[key]])) for l in LABELS}
            lab = max(avg, key=avg.get)
            r = {"box": box, "label": lab, "prob": avg[lab], "probs": avg}
            if self.check_quality:
                r["quality"] = analyze_face_quality(frame_bgr, f)
                confidence_warning = low_confidence_warning(r["prob"])
                if confidence_warning:
                    r["quality"]["warnings"].append(confidence_warning)
                    r["quality"]["ok"] = False
            if f["blendshapes"] is not None:
                r["blendshapes"] = f["blendshapes"]
            out.append(r)
        for k in [k for k in self._hist if k not in taken]:
            del self._hist[k]
        return out

    def crop_for_face(self, frame_bgr, face):
        """Dùng đúng cùng một crop cho dự đoán và ảnh xem trước."""
        if self.align:
            return align_crop(frame_bgr, face["box"], face["angle"], self.crop_margin)
        return crop_face(frame_bgr, face["box"], self.crop_margin)

    def get_face_crop(self, frame_bgr, box):
        """Ảnh crop của một kết quả trong khung vừa xử lý, chưa chuẩn hóa màu."""
        face = next((face for face in self._faces if face["box"] == box), None)
        return None if face is None else self.crop_for_face(frame_bgr, face)


_default = Pipeline()


def process_frame(frame_bgr, detector="mediapipe", model=None):
    return _default.process_frame(frame_bgr, detector, model)


# ---------------------------------------------------------------- Webcam
def open_camera(index=0, warmup=40):
    """Mở webcam, thử lần lượt DirectShow, MSMF, mặc định (Windows hay lỗi 'không đọc được khung hình')."""
    for backend in (getattr(cv2, "CAP_DSHOW", 700), getattr(cv2, "CAP_MSMF", 1400), cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            for _ in range(warmup):
                ok, frame = cap.read()
                if ok and frame is not None:
                    return cap
                time.sleep(0.05)
        cap.release()
    return None


class CameraThread:
    """Đọc camera ở luồng riêng để không bị trễ do bộ đệm."""
    def __init__(self, index=0):
        self.cap = open_camera(index)             # thử DirectShow/MSMF/mặc định, chỉ nhận camera đọc được khung thật
        self.ok = self.cap is not None
        self.frame, self.running = None, self.ok
        if self.ok:
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                self.frame = f

    def read(self):
        return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        time.sleep(0.05)
        self.cap.release()


def main():
    cam = CameraThread(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    if not cam.ok:
        sys.exit("Không mở được camera (kiểm tra webcam / quyền truy cập camera).")
    detector, model = "mediapipe", DEFAULT_MODEL
    print("MediaPipe + EfficientNet | a = căn chỉnh bật/tắt | q = thoát")
    pipe, fps, prev = Pipeline(), 0.0, time.time()
    mirror = True     # lật gương như soi gương (đặt False nếu không cần)
    while True:
        frame = cam.read()
        if frame is None:
            time.sleep(0.01)
            continue
        if mirror:
            frame = cv2.flip(frame, 1)
        for r in pipe.process_frame(frame, detector, model):
            x, y, w, h = r["box"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(frame, f'{r["label"]} {r["prob"]:.0%}', (x, max(y - 8, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6) if fps else 1 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(frame, f"FPS {fps:.1f} | {detector} | {model} | align {'on' if pipe.align else 'off'}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, "MediaPipe + EfficientNet | a: align | q: quit", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.imshow("Emotion realtime", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("a"):
            pipe.align = not pipe.align
    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
