"""
condition_test.py – ghi log kiểm thử điều kiện thực tế (mục 6)

Mỗi lần chạy = 1 người thử × 1 biểu cảm × 1 điều kiện, ghi ~5 giây, mỗi khung 1 dòng
vào conditions_log.csv (nối thêm vào file cũ).

Ví dụ:
    python condition_test.py --who An --expr happy --light binh_thuong --angle 0 --dist 1
    python condition_test.py --who An --expr fear  --light thieu_sang  --angle 0 --dist 1 \

Giá trị gợi ý:
    --light  sang_deu | binh_thuong | thieu_sang | nguoc_sang
    --angle  0 | 30 | 45            (độ nghiêng mặt)
    --dist   0.5 | 1 | 2 | 3        (mét)
    --occl   none | kinh | khau_trang
    --no-align  tắt căn chỉnh mặt bằng landmark (chỉ có tác dụng với mediapipe; dùng để so sánh có/không căn chỉnh)
    --expr   surprise fear disgust happy sad angry neutral (biểu cảm người thử được nói trước)

Chỉ lưu SỐ LIỆU (không lưu hình). Chỉ quay người đã đồng ý.
Phím: q = hủy lần ghi này (không ghi gì).
"""
import argparse
import csv
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "core"))
_HERE = os.path.join(_ROOT, "results")
os.makedirs(_HERE, exist_ok=True)

import cv2
from pipeline import Pipeline
import detectors

LABELS = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]
FIELDS = ["time_s", "who", "expr_true", "light", "angle", "dist", "occl", "detector", "model", "align",
          "face_detected", "n_faces", "label_pred", "prob", "correct"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--who", required=True)
    ap.add_argument("--expr", required=True, choices=LABELS)
    ap.add_argument("--light", default="binh_thuong")
    ap.add_argument("--angle", default="0")
    ap.add_argument("--dist", default="1")
    ap.add_argument("--occl", default="none")
    ap.add_argument("--no-align", action="store_true", help="tắt căn chỉnh mặt (để so sánh)")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--countdown", type=float, default=3.0)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(_HERE, "conditions_log.csv"))
    a = ap.parse_args()
    a.detector = "mediapipe"
    a.model = "efficientnet"

    from pipeline import open_camera
    cap = open_camera(a.cam)
    if cap is None:
        sys.exit("Không mở được camera.")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    pipe = Pipeline(detect_every=1, align=not a.no_align)       # kiểm thử độ bền: phát hiện mọi khung

    # Làm nóng TRƯỚC khi đếm giờ: lần đầu nạp model/detector mất vài giây, không được ăn vào thời gian ghi
    frame = None
    for _ in range(30):
        ok, frame = cap.read()
        if ok:
            break
    if frame is None:
        sys.exit("Camera không trả về khung hình.")
    pipe.process_frame(frame, detector=a.detector, model=a.model)
    if detectors._lm_failed:
        sys.exit("MediaPipe không khởi tạo được (thiếu face_landmarker.task / không có mạng?). "
                 "Không ghi log để tránh số liệu sai.")
    pipe.reset()

    rows, t_start = [], time.time()
    t_rec = t_start + a.countdown
    cancelled = False

    while time.time() < t_rec + a.seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        now = time.time()
        res = pipe.process_frame(frame, detector=a.detector, model=a.model)  # chạy cả lúc đếm ngược để làm nóng + làm mượt
        r = max(res, key=lambda r: r["box"][2] * r["box"][3]) if res else None

        if now < t_rec:
            msg, col = f"CHUAN BI: {a.expr.upper()} sau {t_rec - now:.0f}s", (0, 200, 255)
        else:
            msg, col = f"DANG GHI: {a.expr.upper()}  ({t_rec + a.seconds - now:.1f}s)", (0, 0, 255)
            rows.append({
                "time_s": round(now - t_rec, 3), "who": a.who, "expr_true": a.expr,
                "light": a.light, "angle": a.angle, "dist": a.dist, "occl": a.occl,
                "detector": a.detector, "model": a.model, "align": "off" if a.no_align else "on",
                "face_detected": int(r is not None), "n_faces": len(res),
                "label_pred": r["label"] if r else "",
                "prob": round(r["prob"], 3) if r else "",
                "correct": int(r["label"] == a.expr) if r else "",
            })
        if r:
            x, y, w, h = r["box"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(frame, f'{r["label"]} {r["prob"]:.0%}', (x, max(18, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        cv2.imshow("Condition test (q = huy)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cancelled = True
            break

    cap.release()
    cv2.destroyAllWindows()
    if cancelled or not rows:
        print("Đã hủy / không có khung nào — không ghi.")
        return
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
    det = sum(r["face_detected"] for r in rows)
    ok = sum(r["correct"] == 1 for r in rows)
    print(f"Đã ghi {len(rows)} khung vào {a.out}: có mặt {det}, đoán đúng {ok}")


if __name__ == "__main__":
    main()
