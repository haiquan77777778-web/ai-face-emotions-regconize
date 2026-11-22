"""Đánh giá chất lượng ảnh đầu vào bằng các phép đo nhẹ, không phụ thuộc mô hình cảm xúc."""

import cv2
import numpy as np


LOW_CONFIDENCE = 0.55


def _warning(code, title, guide, severity="warning"):
    return {"code": code, "title": title, "guide": guide, "severity": severity}


def _gray_metrics(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Phần dư sau lọc trung vị là một ước lượng đơn giản cho nhiễu cao tần.
    residual = gray.astype(np.float32) - cv2.medianBlur(gray, 3).astype(np.float32)
    noise = float(residual.std())
    return {"brightness": brightness, "blur": blur, "noise": noise}


def analyze_frame_quality(frame_bgr):
    """Trả cảnh báo áp dụng cho toàn ảnh cùng các chỉ số phục vụ hiển thị/debug."""
    if frame_bgr is None or frame_bgr.size == 0:
        return {"ok": False, "metrics": {}, "warnings": [
            _warning("invalid_image", "Ảnh không hợp lệ", "Chọn lại một ảnh JPG, PNG hoặc WebP hợp lệ.", "error")
        ]}

    metrics = _gray_metrics(frame_bgr)
    warnings = []
    if metrics["brightness"] < 55:
        warnings.append(_warning("too_dark", "Hình ảnh quá tối", "Tăng ánh sáng phía trước mặt hoặc di chuyển đến nơi sáng hơn."))
    elif metrics["brightness"] > 210:
        warnings.append(_warning("too_bright", "Hình ảnh quá sáng", "Giảm nguồn sáng trực tiếp và tránh đứng ngược sáng."))
    if metrics["blur"] < 45:
        warnings.append(_warning("blurred", "Hình ảnh bị mờ", "Giữ camera ổn định, lau ống kính và chờ camera lấy nét."))
    if metrics["noise"] > 22:
        warnings.append(_warning("noisy", "Hình ảnh có nhiều nhiễu", "Tăng ánh sáng môi trường và dùng camera có độ phân giải cao hơn."))
    return {"ok": not warnings, "metrics": metrics, "warnings": warnings}


def analyze_face_quality(frame_bgr, face):
    """Đánh giá vị trí, kích thước, tư thế và chất lượng vùng của một khuôn mặt."""
    H, W = frame_bgr.shape[:2]
    x, y, w, h = [int(v) for v in face["box"]]
    warnings = []
    area_ratio = (w * h) / max(W * H, 1)
    size_ratio = max(w / max(W, 1), h / max(H, 1))

    if area_ratio < 0.025 or min(w, h) < 60:
        warnings.append(_warning("too_far", "Khuôn mặt quá xa camera", "Di chuyển lại gần để khuôn mặt chiếm phần lớn hơn trong khung hình."))
    elif area_ratio > 0.52 or size_ratio > 0.85:
        warnings.append(_warning("too_near", "Khuôn mặt quá gần camera", "Lùi ra xa để thấy trọn khuôn mặt và một phần vai."))

    edge = max(3, int(min(W, H) * 0.01))
    if x <= edge or y <= edge or x + w >= W - edge or y + h >= H - edge:
        warnings.append(_warning(
            "face_clipped", "Khuôn mặt bị che khuất hoặc nằm ngoài khung hình",
            "Đưa toàn bộ khuôn mặt vào giữa khung hình và bỏ vật che trước mắt, mũi, miệng."))

    pose = face.get("pose") or {}
    roll = float(pose.get("roll", face.get("angle", 0.0)))
    yaw = float(pose.get("yaw", 0.0))
    pitch = float(pose.get("pitch", 0.0))
    if abs(roll) > 15:
        warnings.append(_warning("head_tilt", "Khuôn mặt đang nghiêng", "Giữ đầu thẳng để hai mắt nằm ngang."))
    if abs(yaw) > 18:
        direction = "trái" if yaw < 0 else "phải"
        warnings.append(_warning("head_turn", f"Khuôn mặt quay sang {direction}", "Nhìn thẳng vào ống kính camera."))
    if pitch > 16:
        warnings.append(_warning("head_down", "Khuôn mặt đang cúi", "Nâng đầu và nhìn thẳng vào camera."))
    elif pitch < -16:
        warnings.append(_warning("head_up", "Khuôn mặt đang ngẩng", "Hạ cằm và nhìn thẳng vào camera."))

    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    crop = frame_bgr[y0:y1, x0:x1]
    metrics = _gray_metrics(crop) if crop.size else {}
    if metrics:
        if metrics["brightness"] < 50:
            warnings.append(_warning("face_too_dark", "Vùng khuôn mặt quá tối", "Chiếu sáng đều từ phía trước khuôn mặt."))
        elif metrics["brightness"] > 215:
            warnings.append(_warning("face_too_bright", "Vùng khuôn mặt quá sáng", "Giảm ánh sáng trực tiếp chiếu vào khuôn mặt."))
        if metrics["blur"] < 35:
            warnings.append(_warning("face_blurred", "Khuôn mặt bị mờ", "Giữ camera ổn định và chờ camera lấy nét vào khuôn mặt."))
        if metrics["noise"] > 22:
            warnings.append(_warning("face_noisy", "Vùng khuôn mặt có nhiều nhiễu", "Tăng ánh sáng và dùng ảnh có độ phân giải tốt hơn."))
    return {
        "ok": not warnings,
        "metrics": {"area_ratio": area_ratio, "roll": roll, "yaw": yaw, "pitch": pitch, **metrics},
        "warnings": warnings,
    }


def low_confidence_warning(probability, threshold=LOW_CONFIDENCE):
    if float(probability) >= threshold:
        return None
    return _warning(
        "low_confidence", "Kết quả có độ tin cậy thấp",
        "Điều chỉnh theo các cảnh báo chất lượng, nhìn thẳng camera rồi thử lại.")
