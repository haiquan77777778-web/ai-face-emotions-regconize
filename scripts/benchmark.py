"""
benchmark.py – đo FPS và latency (chạy 300 khung, bỏ 30 khung đầu để khởi động)

Đo MediaPipe + EfficientNet trên một chuỗi khung hình đọc từ webcam. Đo riêng thời gian phát hiện,
thời gian phân loại (đã gồm căn chỉnh mặt), FPS tổng và độ trễ p95 mỗi khung. Ghi benchmark.csv (kèm cấu hình máy).

    python scripts/benchmark.py        (ngồi trước webcam, giữ mặt trong khung)
"""
import csv
import os
import platform
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))

import cv2
import numpy as np

from detectors import detect_faces, align_crop
from predict import predict, DEFAULT_MODEL

WARMUP, MEASURE, DETECT_WIDTH = 30, 300, 480
OUT = os.path.join(ROOT, "results", "benchmark.csv")


def machine_info():
    info = f"{platform.processor() or platform.machine()} | {os.cpu_count()} luồng | Python {platform.python_version()}"
    try:
        import psutil
        info += f" | RAM {psutil.virtual_memory().total // 1024 ** 3}GB"
    except ImportError:
        pass
    gpu = "không (CPU)"
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        try:
            import tensorflow as tf
            g = tf.config.list_physical_devices("GPU")
            gpu = g[0].name if g else gpu
        except ImportError:
            pass
    info += f" | GPU: {gpu}"
    return info


def read_frames(n):
    """Đọc n khung hình từ webcam (một lần), dùng chung cho mọi tổ hợp."""
    from pipeline import open_camera
    cap = open_camera(0)
    if cap is None:
        sys.exit("Không mở được webcam.")
    frames = []
    while len(frames) < n:
        ok, f = cap.read()
        if ok:
            frames.append(cv2.flip(f, 1))
    cap.release()
    return frames


def run(frames, det, mdl):
    td, tc, tt, nface = [], [], [], 0
    t_all = time.perf_counter()
    for f in frames:
        H, W = f.shape[:2]
        s = DETECT_WIDTH / W if W > DETECT_WIDTH else 1.0
        small = cv2.resize(f, (int(W * s), int(H * s))) if s < 1 else f
        t0 = time.perf_counter()
        faces = detect_faces(small, det)
        t1 = time.perf_counter()
        td.append(t1 - t0)
        if faces:
            box = tuple(int(v / s) for v in faces[0]["box"])
            face = align_crop(f, box, faces[0]["angle"])
            if face.shape[0] >= 20 and face.shape[1] >= 20:
                predict(face, mdl)
                nface += 1
        t2 = time.perf_counter()
        tc.append(t2 - t1)
        tt.append(t2 - t0)
    fps = len(frames) / (time.perf_counter() - t_all)
    return np.mean(td) * 1000, np.mean(tc) * 1000, fps, 100 * nface / len(frames), np.percentile(tt, 95) * 1000


def main():
    frames = read_frames(WARMUP + MEASURE)
    machine, rows = machine_info(), []
    print("Máy:", machine)
    det, mdl = "mediapipe", DEFAULT_MODEL
    print(f"\n{det} + {mdl}: khởi động {WARMUP} khung, đo {MEASURE} khung ...")
    run(frames[:WARMUP], det, mdl)
    ms_d, ms_c, fps, face, p95 = run(frames[WARMUP:], det, mdl)
    if face == 0:
        sys.exit("Không khung nào có mặt.")
    print(f"  detect {ms_d:.1f} ms | classify {ms_c:.1f} ms | {fps:.1f} FPS | p95 {p95:.1f} ms | có mặt {face:.0f}% khung")
    rows.append({"detector": det, "model": mdl, "ms_detect": round(ms_d, 2),
                 "ms_classify": round(ms_c, 2), "fps": round(fps, 2), "ms_p95": round(p95, 2),
                 "face_pct": round(face, 1), "machine": machine})
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["detector", "model", "ms_detect", "ms_classify", "fps", "ms_p95", "face_pct", "machine"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nĐã lưu {OUT}")


if __name__ == "__main__":
    main()
