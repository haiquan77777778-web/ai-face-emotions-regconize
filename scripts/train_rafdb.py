"""Fine-tune EfficientNet đã có bằng RAF-DB, cân bằng lớp và chọn theo validation macro-F1.

python scripts/train_rafdb.py --data-dir D:/datasets/RAF-DB --dry-run
python scripts/train_rafdb.py --data-dir D:/datasets/RAF-DB --epochs 15

Hỗ trợ bản gốc (list_patition_label.txt + Image/aligned) hoặc thư mục
train/<tên lớp hoặc 1..7>/*.jpg, val/... (tùy chọn), test/... (tùy chọn).
"""
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
from preprocessing import prepare_face

LABELS = (ROOT / "models" / "labels.txt").read_text(encoding="utf-8").split()
RAF_LABELS = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def folder_samples(directory):
    samples = []
    for folder in sorted(directory.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name.lower()
        if name.isdigit() and 1 <= int(name) <= 7:
            name = RAF_LABELS[int(name) - 1]
        name = {"anger": "angry", "happiness": "happy", "sadness": "sad"}.get(name, name)
        images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if images and name not in LABELS:
            raise ValueError(f"Thư mục lớp không hợp lệ: {folder}")
        samples.extend((p.resolve(), LABELS.index(name)) for p in images)
    return samples


def load_samples(data_dir, labels_file=None, images_dir=None):
    """Giữ nguyên tập test chính thức; không đưa ảnh test vào train/validation."""
    if not data_dir.is_dir():
        raise ValueError(f"Không tìm thấy thư mục dữ liệu: {data_dir}")
    if labels_file is None and (data_dir / "train").is_dir():
        train = folder_samples(data_dir / "train")
        val_dirs = [data_dir / n for n in ("val", "validation") if (data_dir / n).is_dir()]
        if len(val_dirs) > 1:
            raise ValueError("Chỉ giữ một thư mục val hoặc validation để tránh chọn nhầm dữ liệu.")
        val = folder_samples(val_dirs[0]) if val_dirs else []
        if val_dirs and not val:
            raise ValueError(f"Thư mục validation không có ảnh: {val_dirs[0]}")
        test = folder_samples(data_dir / "test") if (data_dir / "test").is_dir() else []
        return train, val, test

    if labels_file is None:
        candidates = sorted({*data_dir.rglob("list_patition_label.txt"),
                             *data_dir.rglob("list_partition_label.txt")})
        if len(candidates) != 1:
            raise ValueError("Cần đúng một file nhãn RAF-DB; chỉ rõ bằng --labels-file.")
        labels_file = candidates[0]
    if images_dir is None:
        candidates = sorted(p for p in data_dir.rglob("aligned") if p.is_dir())
        if len(candidates) != 1:
            raise ValueError("Cần thư mục ảnh aligned; chỉ rõ bằng --images-dir.")
        images_dir = candidates[0]
    splits = {"train": [], "test": []}
    seen = set()
    for line in labels_file.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        filename, raw_label = line.split()
        label = int(raw_label)
        if not 1 <= label <= 7:
            raise ValueError(f"Nhãn RAF-DB không hợp lệ: {line}")
        image = Path(filename)
        aligned_name = image.stem if image.stem.endswith("_aligned") else image.stem + "_aligned"
        path = (images_dir / (aligned_name + image.suffix)).resolve()
        if not path.is_file():
            raise ValueError(f"Thiếu ảnh đã căn chỉnh: {path}")
        split = "train" if image.name.startswith("train_") else "test" if image.name.startswith("test_") else None
        if split is None or path in seen:
            raise ValueError(f"Tên ảnh không hợp lệ hoặc bị trùng: {filename}")
        seen.add(path)
        splits[split].append((path, LABELS.index(RAF_LABELS[label - 1])))
    return splits["train"], [], splits["test"]


def split_validation(samples, fraction, seed):
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in range(len(LABELS)):
        group = [sample for sample in samples if sample[1] == label]
        if len(group) < 2:
            raise ValueError(f"Lớp {LABELS[label]} cần ít nhất 2 ảnh train để chia validation.")
        rng.shuffle(group)
        n = min(len(group) - 1, max(1, round(len(group) * fraction)))
        val.extend(group[:n])
        train.extend(group[n:])
    return train, val


def balanced_weights(samples):
    counts = np.bincount([label for _, label in samples], minlength=len(LABELS))
    if np.any(counts == 0):
        raise ValueError("Tập train phải có đủ 7 lớp.")
    return {i: float(len(samples) / (len(LABELS) * count)) for i, count in enumerate(counts)}


def check_splits(splits):
    seen = set()
    for name, samples in splits.items():
        paths = {p for p, _ in samples}
        if len(paths) != len(samples) or paths & seen:
            raise ValueError(f"Có ảnh trùng đường dẫn giữa các tập: {name}")
        seen.update(paths)
        if name in ("train", "validation") and {y for _, y in samples} != set(range(len(LABELS))):
            raise ValueError(f"Tập {name} phải có đủ 7 lớp.")


def scores_from_confusion(cm):
    tp = np.diag(cm)
    precision = np.divide(tp, cm.sum(axis=0), out=np.zeros(len(LABELS)), where=cm.sum(axis=0) != 0)
    recall = np.divide(tp, cm.sum(axis=1), out=np.zeros(len(LABELS)), where=cm.sum(axis=1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(len(LABELS)), where=precision + recall != 0)
    return {"accuracy": float(tp.sum() / cm.sum()) if cm.sum() else 0.0,
            "macro_f1": float(f1.mean()), "per_class": {
                label: {"precision": float(precision[i]), "recall": float(recall[i]),
                        "f1": float(f1[i]), "support": int(cm[i].sum())}
                for i, label in enumerate(LABELS)}, "confusion_matrix": cm.tolist()}


def make_dataset(tf, samples, batch_size, training=False, seed=42):
    paths = [str(p) for p, _ in samples]
    labels = [label for _, label in samples]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(samples), seed=seed, reshuffle_each_iteration=True)

    def read_image(path, label):
        def decode(value):
            filename = value.numpy().decode("utf-8")
            image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Ảnh không đọc được: {filename}")
            return prepare_face(image)
        image = tf.py_function(decode, [path], tf.float32)
        image.set_shape((100, 100, 3))
        return image, label

    # Giới hạn luồng để không tranh tài nguyên với TensorFlow trên máy CPU.
    return ds.map(read_image, num_parallel_calls=2).batch(batch_size).prefetch(1)


def evaluate(model, dataset):
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for images, labels in dataset:
        prediction = model(images, training=False).numpy().argmax(axis=1)
        np.add.at(cm, (labels.numpy(), prediction), 1)
    return scores_from_confusion(cm)


def configure_fine_tuning(model, keras, unfreeze_layers):
    """Mở các lớp có trọng số ở cuối mạng; giữ BatchNormalization đóng băng."""
    leaves = []

    def visit(layer):
        layer.trainable = True
        if isinstance(layer, keras.Model):
            for child in layer.layers:
                visit(child)
        else:
            leaves.append(layer)

    visit(model)
    candidates = [layer for layer in leaves if layer.weights
                  and not isinstance(layer, (keras.layers.BatchNormalization, keras.layers.Normalization))]
    selected = {id(layer) for layer in candidates[-unfreeze_layers:]}
    for layer in leaves:
        layer.trainable = id(layer) in selected
    if not model.trainable_weights:
        raise ValueError("Không có lớp nào để fine-tune.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--labels-file", type=Path)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--initial-model", type=Path, default=ROOT / "models" / "efficientnet.keras")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "efficientnet_rafdb.keras")
    parser.add_argument("--report-dir", type=Path, help="Thư mục mới để lưu báo cáo; mặc định results/training/<thời_gian>.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--unfreeze-layers", type=int, default=30)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--evaluate-test", action="store_true", help="Đánh giá test một lần sau khi chọn model bằng validation.")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ kiểm tra tập dữ liệu và class_weight, không train.")
    args = parser.parse_args(argv)
    if not 0 < args.val_fraction < 1 or min(args.epochs, args.batch_size, args.unfreeze_layers, args.patience) < 1:
        parser.error("val-fraction phải trong (0,1); epochs/batch-size/unfreeze-layers/patience phải >= 1.")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate phải là số hữu hạn > 0.")
    if set(LABELS) != set(RAF_LABELS) or len(LABELS) != 7:
        parser.error("models/labels.txt phải chứa đúng 7 nhãn RAF-DB.")
    try:
        train, val, test = load_samples(args.data_dir, args.labels_file, args.images_dir)
        if not val:
            train, val = split_validation(train, args.val_fraction, args.seed)
        splits = {"train": train, "validation": val, "test": test}
        check_splits(splits)
        weights = balanced_weights(train)
        if args.evaluate_test and not test:
            raise ValueError("Không có tập test riêng để --evaluate-test.")
        print("label          train    val   test    class_weight")
        for i, label in enumerate(LABELS):
            counts = [sum(y == i for _, y in values) for values in splits.values()]
            print(f"{label:12s} {counts[0]:6d} {counts[1]:6d} {counts[2]:6d}    {weights[i]:.3f}")
        if args.dry_run:
            print("Dữ liệu hợp lệ. Chưa đọc nội dung ảnh hoặc kiểm tra chất lượng nhãn; chưa huấn luyện.")
            return 0
        if args.output.suffix != ".keras" or args.output.exists():
            raise ValueError("--output phải là file .keras mới; chọn tên khác nếu file đã tồn tại.")
        if args.report_dir is not None and args.report_dir.exists():
            raise ValueError("--report-dir phải là thư mục mới.")
        if not args.initial_model.is_file():
            raise ValueError(f"Không tìm thấy model ban đầu: {args.initial_model}")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    import tensorflow as tf
    from tensorflow import keras

    keras.utils.set_random_seed(args.seed)
    model = keras.models.load_model(args.initial_model, compile=False)
    if tuple(model.input_shape[1:]) != (100, 100, 3) or model.output_shape[-1] != len(LABELS):
        parser.error("Model phải nhận RGB (100,100,3), giá trị 0..255 và trả 7 xác suất softmax.")
    train_ds = make_dataset(tf, train, args.batch_size, training=True, seed=args.seed)
    val_ds = make_dataset(tf, val, args.batch_size)
    baseline = evaluate(model, val_ds)
    configure_fine_tuning(model, keras, args.unfreeze_layers)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_dir = args.report_dir or ROOT / "results" / "training" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    report_dir.mkdir(parents=True)
    with (report_dir / "splits.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "path", "label"])
        for split, samples in splits.items():
            writer.writerows((split, str(path), LABELS[label]) for path, label in samples)
    run = {"config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
           "labels": LABELS, "class_weight": weights, "baseline_validation": baseline,
           "status": "training", "best_epoch": 0, "best_macro_f1": baseline["macro_f1"],
           "validation_scope": "Held out from this fine-tune only; initial model training data is unknown."}

    def write_report():
        (report_dir / "report.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")

    class MacroF1Checkpoint(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best_score = baseline["macro_f1"]
            self.best_epoch = 0

        def on_epoch_end(self, epoch, logs=None):
            scores = evaluate(self.model, val_ds)
            logs["val_macro_f1"] = scores["macro_f1"]
            for label in ("fear", "disgust"):
                logs[f"val_{label}_f1"] = scores["per_class"][label]["f1"]
            print(f"\nValidation macro-F1={scores['macro_f1']:.4f}; "
                  f"fear={logs['val_fear_f1']:.4f}; disgust={logs['val_disgust_f1']:.4f}")
            if scores["macro_f1"] > self.best_score:
                self.best_score = scores["macro_f1"]
                self.best_epoch = epoch + 1
                self.model.save(args.output)
            run["best_epoch"] = self.best_epoch
            run["best_macro_f1"] = self.best_score
            write_report()

    # Giữ baseline nếu mọi epoch mới đều kém hơn trên validation macro-F1.
    model.save(args.output)
    write_report()
    print(f"Baseline validation macro-F1: {baseline['macro_f1']:.4f}")
    checkpoint = MacroF1Checkpoint()
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, class_weight=weights,
              callbacks=[checkpoint,
                         keras.callbacks.CSVLogger(str(report_dir / "history.csv")),
                         keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=args.patience)])
    best = keras.models.load_model(args.output, compile=False)
    run["best_validation"] = evaluate(best, val_ds)
    if args.evaluate_test:
        run["test"] = evaluate(best, make_dataset(tf, test, args.batch_size))
    run["status"] = "complete"
    write_report()
    rows = []
    for stage, scores in (("baseline", baseline), ("best", run["best_validation"])):
        rows.extend({"stage": stage, "label": label, **metrics} for label, metrics in scores["per_class"].items())
    with (report_dir / "metrics_per_class.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["stage", "label", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Model: {args.output}\nBáo cáo: {report_dir}")
    print("Khởi động lại app để chọn 'EfficientNet (fine-tune RAF-DB)' nếu dùng tên output mặc định.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
