"""So tốc độ eager/graph trên cùng ảnh giả; chỉ đo phân loại, không đo FPS webcam.

python scripts/benchmark_inference.py --model efficientnet --frames 30
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
from predict import DEFAULT_MODEL, MODEL_NAMES, _get, predict
from preprocessing import prepare_face


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, default=DEFAULT_MODEL)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, help="CSV tùy chọn để lưu kết quả.")
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 1:
        parser.error("frames và warmup phải >= 1.")
    model = _get(args.model)
    rng = np.random.default_rng(42)
    faces = [rng.integers(0, 256, (160, 160, 3), dtype=np.uint8)
             for _ in range(max(args.frames, args.warmup))]

    def eager(face):
        return model(prepare_face(face)[None], training=False).numpy()[0]

    def graph(face):
        _, _, probabilities = predict(face, args.model)
        return np.array(list(probabilities.values()))

    # Tách thời gian nạp/trace khỏi thời gian suy luận, kiểm tra trên nhiều ảnh.
    for face in faces[:3]:
        np.testing.assert_allclose(eager(face), graph(face), rtol=1e-4, atol=1e-5)
    rows = []
    for name, function in (("eager", eager), ("graph", graph)):
        for face in faces[:args.warmup]:
            function(face)
        elapsed = []
        for face in faces[:args.frames]:
            start = time.perf_counter()
            function(face)  # .numpy() chờ kết quả thực, không chỉ đo thời gian gửi lệnh.
            elapsed.append((time.perf_counter() - start) * 1000)
        mean = float(np.mean(elapsed))
        row = {"model": args.model, "mode": name, "frames": args.frames,
               "mean_ms": mean, "p95_ms": float(np.percentile(elapsed, 95)),
               "classify_fps": 1000 / mean}
        rows.append(row)
        print(f"{name}: {mean:.2f} ms | p95 {row['p95_ms']:.2f} ms | "
              f"{row['classify_fps']:.1f} lượt phân loại/giây", flush=True)
    print(f"Tăng tốc: {rows[0]['mean_ms'] / rows[1]['mean_ms']:.2f}x. "
          "Đây không phải FPS webcam; số liệu dùng ảnh giả và một khuôn mặt.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
