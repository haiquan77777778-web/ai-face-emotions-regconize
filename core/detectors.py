"""
detectors.py – phát hiện khuôn mặt

Giao tiếp (đúng hợp đồng):
    detect(frame_bgr, detector="mediapipe") -> [(x, y, w, h), ...]   (pixel)
    crop_face(frame, box, margin=0.15)  -> ảnh mặt có lề

Detector cố định: MediaPipe Face Landmarker (khung mặt + landmark + blendshape),
cần face_landmarker.task, tự tải lần đầu nếu chưa có.
detect_faces() trả thêm góc nghiêng (từ hai mắt) và blendshape; align_crop() dùng góc đó để căn mặt thẳng.
Chạy thử webcam:  python detectors.py     (q = thoát)
"""
import os
import sys
import time
import urllib.request

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
LM_MODEL = os.path.join(_HERE, "face_landmarker.task")
LM_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
          "face_landmarker/float16/1/face_landmarker.task")

# ---------------------------------------------------------------- MediaPipe Face Landmarker
# Một lần chạy cho ra: khung mặt + 478 landmark + 52 blendshape.
# Khung mặt lấy từ landmark; landmark hai mắt dùng để căn chỉnh (xoay) mặt cho thẳng.
_lm = None
_lm_failed = False
_lm_ts = 0
EYE_R, EYE_L = 33, 263          # khóe mắt ngoài (bên trái / bên phải ảnh)
NOSE, CHIN, CHEEK_R, CHEEK_L = 1, 152, 234, 454


def _get_lm():
    global _lm, _lm_failed
    if _lm is None and not _lm_failed:
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            if not os.path.exists(LM_MODEL):
                print("[MediaPipe] tải face_landmarker.task ...")
                urllib.request.urlretrieve(LM_URL, LM_MODEL)
            opts = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=LM_MODEL),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=5, output_face_blendshapes=True)
            _lm = vision.FaceLandmarker.create_from_options(opts)
        except Exception as e:
            print(f"[MediaPipe] không khởi tạo được: {e}")
            _lm_failed = True
    return _lm


def _faces_from_result(res, W, H):
    """Đổi kết quả Face Landmarker sang [{"box","angle","blendshapes"}] (pixel)."""
    out = []
    for i, lms in enumerate(res.face_landmarks):
        xs = np.array([p.x for p in lms]) * W
        ys = np.array([p.y for p in lms]) * H
        x0, y0 = max(0, int(xs.min())), max(0, int(ys.min()))
        x1, y1 = min(W, int(xs.max())), min(H, int(ys.max()))
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        # góc nghiêng của đường nối hai mắt (độ); xoay ảnh đúng góc này là mắt nằm ngang
        angle = float(np.degrees(np.arctan2(ys[EYE_L] - ys[EYE_R], xs[EYE_L] - xs[EYE_R])))
        # Ước lượng yaw/pitch từ vị trí tương đối của mũi. Các giá trị chỉ dùng để
        # cảnh báo tư thế, không tham gia dự đoán cảm xúc.
        face_width = max(xs[CHEEK_L] - xs[CHEEK_R], 1.0)
        center_x = (xs[CHEEK_L] + xs[CHEEK_R]) / 2
        yaw = float((xs[NOSE] - center_x) / face_width * 100)
        eye_y = (ys[EYE_R] + ys[EYE_L]) / 2
        eye_to_chin = max(ys[CHIN] - eye_y, 1.0)
        pitch = float(((ys[NOSE] - eye_y) / eye_to_chin - 0.30) * 100)
        bs = None
        if res.face_blendshapes and i < len(res.face_blendshapes):
            bs = {c.category_name: float(c.score) for c in res.face_blendshapes[i]
                  if c.category_name != "_neutral"}
        out.append({"box": (x0, y0, x1 - x0, y1 - y0), "angle": angle,
                    "pose": {"roll": angle, "yaw": yaw, "pitch": pitch},
                    "blendshapes": bs})
    return out


def detect_mediapipe_faces(frame):
    global _lm_ts
    lm = _get_lm()
    if lm is None:
        return []
    import mediapipe as mp
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    _lm_ts += 33
    res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), _lm_ts)
    return _faces_from_result(res, frame.shape[1], frame.shape[0])


# ---------------------------------------------------------------- Hợp đồng
def detect_faces(frame_bgr, detector="mediapipe"):
    """Phát hiện khuôn mặt bằng MediaPipe và trả thêm tư thế/blendshape."""
    if detector != "mediapipe":
        raise ValueError("Hệ thống chỉ hỗ trợ detector 'mediapipe'.")
    return detect_mediapipe_faces(frame_bgr)


def detect(frame_bgr, detector="mediapipe"):
    return [f["box"] for f in detect_faces(frame_bgr, detector)]


def crop_face(frame, box, margin=0.15):
    x, y, w, h = box
    dx, dy = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - dx), max(0, y - dy)
    x1, y1 = min(frame.shape[1], x + w + dx), min(frame.shape[0], y + h + dy)
    return frame[y0:y1, x0:x1]


def align_crop(frame, box, angle, margin=0.15):
    """Crop mặt rồi xoay cho hai mắt nằm ngang (angle lấy từ landmark). angle ~ 0 thì giống crop_face."""
    face = crop_face(frame, box, margin)
    if face.size == 0 or abs(angle) < 1.0:
        return face
    h, w = face.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(face, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------- Webcam thử detector
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Không mở được camera (kiểm tra webcam / quyền truy cập camera).")
    prev = time.time()
    print("MediaPipe | q = thoát")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Không đọc được khung hình.")
            break
        frame = cv2.flip(frame, 1)       # lật gương
        for x, y, w, h in detect(frame):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        now = time.time()
        cv2.putText(frame, f"FPS {1 / max(now - prev, 1e-6):.1f}  mediapipe", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        prev = now
        cv2.imshow("Detector", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
