"""
predict.py – bộ phân loại cảm xúc
face_bgr: ảnh mặt đã crop từ OpenCV (BGR, kích thước bất kỳ). Nhãn theo models/labels.txt:
surprise, fear, disgust, happy, sad, angry, neutral. Đường dẫn models/ tính từ vị trí file này, chạy ở thư mục nào cũng được.
"""
import os

import tensorflow as tf
from tensorflow import keras

from preprocessing import prepare_face

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(_HERE), "models")

with open(os.path.join(MODEL_DIR, "labels.txt"), encoding="utf-8") as f:
    LABELS = [l.strip() for l in f.read().split("\n") if l.strip()]

MODEL_NAMES = ["efficientnet"]
DEFAULT_MODEL = "efficientnet"
_CACHE = {}
_INFERENCE_CACHE = {}


def _get(model_name):
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Model '{model_name}' không tồn tại, chỉ có: {MODEL_NAMES}")
    if model_name not in _CACHE:
        path = os.path.join(MODEL_DIR, f"{model_name}.keras")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Thiếu file model: {path}")
        _CACHE[model_name] = keras.models.load_model(path, compile=False)
    return _CACHE[model_name]


def _get_inference(model_name):
    """Tạo graph một lần cho mỗi model; không thay đổi trọng số hay preprocessing."""
    model = _get(model_name)
    if model_name not in _INFERENCE_CACHE:
        @tf.function(input_signature=[tf.TensorSpec((None, 100, 100, 3), tf.float32)],
                     autograph=False)
        def inference(images):
            return model(images, training=False)

        _INFERENCE_CACHE[model_name] = inference
    return _INFERENCE_CACHE[model_name]


def predict(face_bgr, model_name=DEFAULT_MODEL):
    x = prepare_face(face_bgr)[None]                # (1, 100, 100, 3), giá trị 0-255
    probs = _get_inference(model_name)(x).numpy()[0]
    i = int(probs.argmax())
    return LABELS[i], float(probs[i]), {l: float(p) for l, p in zip(LABELS, probs)}
