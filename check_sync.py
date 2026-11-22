"""
check_sync.py – kiểm tra predict.py (Người 1) có đúng HỢP ĐỒNG chung không, trước khi chạy pipeline/benchmark.

    python check_sync.py            (đặt predict.py + models/ của Người 1 cạnh file này)

Kiểm tra: LABELS đúng thứ tự, model EfficientNet tồn tại,
predict() trả (nhãn, xác suất, dict 7 lớp tổng ~1), rồi đo thời gian một lần dự đoán.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))
import time

import numpy as np

CONTRACT_LABELS = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]
CONTRACT_MODELS = ["efficientnet"]

try:
    from predict import predict, LABELS, MODEL_NAMES
except Exception as e:
    sys.exit(f"[LỖI] không import được predict.py / LABELS / MODEL_NAMES: {e}")

ok = True


def check(cond, msg):
    global ok
    print(("[OK]   " if cond else "[LỖI] ") + msg)
    ok = ok and cond


check(list(LABELS) == CONTRACT_LABELS, f"LABELS đúng thứ tự hợp đồng ({list(LABELS)})")
missing = [m for m in CONTRACT_MODELS if m not in MODEL_NAMES]
check(not missing, f"MODEL_NAMES = {list(MODEL_NAMES)}" + (f" (thiếu: {missing})" if missing else ""))

face = (np.random.rand(160, 160, 3) * 255).astype(np.uint8)       # ảnh giả, chỉ để kiểm tra hình dạng đầu ra
for m in MODEL_NAMES:
    try:
        predict(face, m)                                             # lần đầu nạp model
        t = time.perf_counter()
        lab, p, probs = predict(face, m)
        ms = (time.perf_counter() - t) * 1000
        good = (lab in CONTRACT_LABELS and 0 <= float(p) <= 1 and isinstance(probs, dict)
                and list(probs) == CONTRACT_LABELS and abs(sum(probs.values()) - 1) < 1e-2
                and abs(probs[lab] - float(p)) < 1e-6)
        check(good, f"predict(face, '{m}') -> ({lab}, {float(p):.2f}, dict {len(probs)} lớp) | {ms:.0f} ms")
    except Exception as e:
        check(False, f"predict(face, '{m}') lỗi: {e}")

print("\nĐỒNG BỘ ĐÚNG HỢP ĐỒNG" if ok else "\nCHƯA ĐỒNG BỘ - sửa các dòng [LỖI] rồi chạy lại")
sys.exit(0 if ok else 1)
