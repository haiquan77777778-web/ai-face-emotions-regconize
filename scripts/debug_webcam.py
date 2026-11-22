"""
debug_webcam.py - xem MÔ HÌNH THẤY GÌ khi chạy webcam, để tìm lỗi "luôn ra neutral".

    python debug_webcam.py

Hiện 2 cửa sổ: hình webcam (khung mặt) và ảnh mặt đã crop đúng như đưa vào mô hình (phóng to).
Trên ảnh in top-3 xác suất của EfficientNet. So ảnh crop này với ảnh RAF-DB: khung quá rộng/quá chặt,
tối, nghiêng, mờ... là nguyên nhân thường gặp.

Phím:  a = bật/tắt căn chỉnh MediaPipe
       [ ] = giảm/tăng lề crop (0, 0.05, 0.15, 0.3)   s = lưu ảnh crop vào debug_crops/   q = thoát
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))

import cv2
import numpy as np

from detectors import detect_faces, crop_face, align_crop
from pipeline import open_camera, DETECT_WIDTH
from predict import predict, DEFAULT_MODEL
from preprocessing import prepare_face

MARGINS = [0.0, 0.05, 0.15, 0.3]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "debug_crops")


def top3(probs):
    return sorted(probs.items(), key=lambda kv: -kv[1])[:3]


def put(img, text, y, color=(0, 255, 255)):
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def main():
    cap = open_camera(0)
    if cap is None:
        sys.exit("Không mở được camera (kiểm tra quyền camera, đóng ứng dụng khác đang dùng camera).")
    det, align, mi = "mediapipe", True, 2
    saved = 0
    print(__doc__)
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)                                  # lật gương như pipeline
        clean = frame.copy()                                        # bản KHÔNG vẽ gì, dùng để crop (tránh khung xanh lọt vào ảnh mô hình)
        h_frame, w_frame = clean.shape[:2]
        scale = min(1.0, DETECT_WIDTH / w_frame)
        small_frame = cv2.resize(clean, (int(w_frame * scale), int(h_frame * scale))) if scale < 1 else clean
        faces = [dict(f, box=tuple(int(v / scale) for v in f["box"]))
                 for f in detect_faces(small_frame, det)]
        crop_view = np.zeros((300, 600, 3), np.uint8)
        face_img = None
        if faces:
            f = max(faces, key=lambda f: f["box"][2] * f["box"][3])     # mặt lớn nhất
            x, y, w, h = f["box"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            face_img = (align_crop(clean, f["box"], f["angle"], MARGINS[mi]) if align and det == "mediapipe"
                        else crop_face(clean, f["box"], MARGINS[mi]))
            if face_img.shape[0] >= 20 and face_img.shape[1] >= 20:
                small = cv2.cvtColor(prepare_face(face_img).astype(np.uint8), cv2.COLOR_RGB2BGR)
                crop_view[0:300, 0:300] = cv2.resize(small, (300, 300), interpolation=cv2.INTER_NEAREST)
                _, _, probs = predict(face_img, DEFAULT_MODEL)
                put(crop_view, DEFAULT_MODEL, 20)
                for j, (label, probability) in enumerate(top3(probs)):
                    put(crop_view, f"{label:9s} {probability:5.0%}", 38 + j * 16, (255, 255, 255))
                cv2.rectangle(crop_view, (310, 25), (310 + int(280 * top3(probs)[0][1]), 35), (0, 200, 0), -1)
        put(frame, f"MediaPipe  align={align}  margin={MARGINS[mi]}  (a, [ ], s, q)", 20)
        cv2.imshow("webcam", frame)
        cv2.imshow("anh dua vao mo hinh (100x100 phong to) + top3", crop_view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("a"):
            align = not align
        elif k == ord("[") and mi > 0:
            mi -= 1
        elif k == ord("]") and mi < len(MARGINS) - 1:
            mi += 1
        elif k == ord("s") and face_img is not None:
            os.makedirs(OUT, exist_ok=True)
            p = os.path.join(OUT, f"crop_{int(time.time())}_{saved}.png")
            cv2.imwrite(p, cv2.resize(face_img, (100, 100), interpolation=cv2.INTER_AREA))
            saved += 1
            print("đã lưu", p)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
