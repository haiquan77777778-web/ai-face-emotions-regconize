"""Tiền xử lý dùng chung cho webcam, debug và fine-tune RAF-DB."""
import cv2

INPUT_SIZE = (100, 100)


def prepare_face(face_bgr):
    """BGR -> RGB float32 0..255; model đã có lớp Rescaling bên trong."""
    if face_bgr is None or face_bgr.size == 0:
        raise ValueError("Ảnh khuôn mặt rỗng.")
    if face_bgr.ndim != 3 or face_bgr.shape[2] != 3:
        raise ValueError("Ảnh khuôn mặt phải có 3 kênh BGR.")
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, INPUT_SIZE, interpolation=cv2.INTER_AREA).astype("float32")
