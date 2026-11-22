"""Hàm thuần (không phụ thuộc Streamlit) cho app.py: lọc, vẽ khung, chuẩn hóa kết quả."""
import time

import cv2

LABELS = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]   # thứ tự hợp đồng


def filter_results(results, threshold):
    """Giữ kết quả có prob >= threshold (0..1)."""
    return [r for r in (results or []) if float(r.get("prob", 0.0)) >= threshold]


def draw_results(frame_bgr, results):
    """Vẽ khung và nhãn lên bản sao của ảnh (pipeline không tự vẽ)."""
    out = frame_bgr.copy()
    for r in results:
        x, y, w, h = [int(v) for v in r["box"]]
        color = (0, 200, 0) if (r.get("quality") or {}).get("ok", True) else (0, 165, 255)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(out, f'{r["label"]} {r["prob"]:.0%}', (x, max(y - 8, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return out


def open_camera(index=0, warmup=40):
    """Mở webcam, thử lần lượt DirectShow, MSMF, mặc định (Windows hay lỗi 'không đọc được khung hình' với backend mặc định).
    Chỉ trả về camera khi ĐỌC ĐƯỢC khung hình thật; trả None nếu không backend nào chạy."""
    for backend in (getattr(cv2, "CAP_DSHOW", 700), getattr(cv2, "CAP_MSMF", 1400), cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            for _ in range(warmup):                 # vài khung đầu thường rỗng, chờ camera khởi động
                ok, frame = cap.read()
                if ok and frame is not None:
                    return cap
                time.sleep(0.05)
        cap.release()
    return None
