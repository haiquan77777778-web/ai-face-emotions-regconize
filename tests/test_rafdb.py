"""Kiểm thử chia dữ liệu, ánh xạ nhãn, class_weight và fine-tune mà không cần RAF-DB thật."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "core"))
from preprocessing import prepare_face
from scripts import train_rafdb as training


class RAFDBTests(unittest.TestCase):
    def test_preprocessing_preserves_rgb_range(self):
        image = np.full((20, 30, 3), (10, 80, 240), dtype=np.uint8)
        result = prepare_face(image)
        self.assertEqual(result.shape, (100, 100, 3))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result[0, 0], (240, 80, 10))
        with self.assertRaises(ValueError):
            prepare_face(np.zeros((0, 0, 3), np.uint8))

    def test_graph_inference_matches_eager_and_reuses_trace(self):
        import predict as classifier
        model = classifier._get("efficientnet")
        rng = np.random.default_rng(15)
        faces = [np.zeros((120, 160, 3), np.uint8),
                 np.full((160, 120, 3), 255, np.uint8),
                 rng.integers(0, 256, (145, 155, 3), dtype=np.uint8)]
        inference = classifier._get_inference("efficientnet")
        for face in faces:
            expected = model(prepare_face(face)[None], training=False).numpy()[0]
            label, probability, probs = classifier.predict(face, "efficientnet")
            np.testing.assert_allclose(list(probs.values()), expected, rtol=1e-4, atol=1e-5)
            self.assertEqual(label, classifier.LABELS[int(expected.argmax())])
            self.assertEqual(probability, probs[label])
        self.assertIs(inference, classifier._get_inference("efficientnet"))
        self.assertEqual(inference.experimental_get_tracing_count(), 1)

    def test_weights_and_stratified_split(self):
        samples = [(Path(f"{label}_{i}.jpg"), label)
                   for label in range(7) for i in range(3 + label)]
        train, val = training.split_validation(samples, 0.2, 42)
        self.assertEqual((train, val), training.split_validation(samples, 0.2, 42))
        training.check_splits({"train": train, "validation": val, "test": []})
        self.assertEqual(len(train) + len(val), len(samples))
        weights = training.balanced_weights(train)
        contributions = [sum(y == i for _, y in train) * weights[i] for i in range(7)]
        np.testing.assert_allclose(contributions, [len(train) / 7] * 7)
        with self.assertRaises(ValueError):
            training.balanced_weights([(Path("one.jpg"), 0)])
        with self.assertRaises(ValueError):
            training.check_splits({"train": train, "validation": train})

    def test_official_label_mapping_and_test_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aligned = root / "Image" / "aligned"
            aligned.mkdir(parents=True)
            labels_file = root / "list_patition_label.txt"
            lines = []
            for prefix in ("train", "test"):
                for label in range(1, 8):
                    name = f"{prefix}_{label:05d}"
                    (aligned / f"{name}_aligned.jpg").touch()
                    lines.append(f"{name}.jpg {label}")
            labels_file.write_text("\n".join(lines), encoding="utf-8")
            train, val, test = training.load_samples(root)
            self.assertFalse(val)
            self.assertEqual([label for _, label in train], list(range(7)))
            self.assertEqual(len(test), 7)
            self.assertTrue(all(p.name.startswith("test_") for p, _ in test))
            labels_file.write_text("train_00001.jpg 0", encoding="utf-8")
            with self.assertRaises(ValueError):
                training.load_samples(root)

    def test_numeric_folders_map_to_contract_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label in range(1, 8):
                directory = root / "train" / str(label)
                directory.mkdir(parents=True)
                (directory / "face.jpg").touch()
            train, _, _ = training.load_samples(root)
            self.assertEqual([label for _, label in train], list(range(7)))

    def test_macro_f1_includes_missed_classes(self):
        cm = np.zeros((7, 7), dtype=int)
        cm[1, 1], cm[1, 2], cm[2, 2], cm[2, 1] = 3, 1, 2, 2
        scores = training.scores_from_confusion(cm)
        self.assertAlmostEqual(scores["per_class"]["fear"]["precision"], 3 / 5)
        self.assertAlmostEqual(scores["per_class"]["fear"]["recall"], 3 / 4)
        self.assertAlmostEqual(scores["accuracy"], 5 / 8)
        self.assertAlmostEqual(scores["macro_f1"], (2 / 3 + 4 / 7) / 7)

    def test_pipeline_resets_smoothing_after_crop_change(self):
        import pipeline
        frame = np.full((120, 120, 3), 100, np.uint8)
        face = {"box": (20, 20, 60, 60), "angle": 0, "blendshapes": None}
        fear = {label: float(label == "fear") for label in training.LABELS}
        disgust = {label: float(label == "disgust") for label in training.LABELS}
        pipe = pipeline.Pipeline(smooth_n=3)
        with patch.object(pipeline, "detect_faces", return_value=[face]), \
             patch.object(pipeline, "predict", side_effect=[("fear", 1, fear), ("disgust", 1, disgust)]) as predict:
            pipe.process_frame(frame)
            self.assertEqual(predict.call_args.args[1], "efficientnet")
            self.assertEqual(pipe.get_face_crop(frame, face["box"]).shape, (78, 78, 3))
            pipe.crop_margin = 0
            result = pipe.process_frame(frame)
            self.assertEqual(result[0]["probs"]["fear"], 0)
            self.assertEqual(result[0]["label"], "disgust")
            self.assertEqual(pipe.get_face_crop(frame, face["box"]).shape, (60, 60, 3))

    def test_training_smoke_and_output_protection(self):
        from tensorflow import keras
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "dữ liệu"
            for split in ("train", "val", "test"):
                for label in training.LABELS:
                    folder = data / split / label
                    folder.mkdir(parents=True)
                    for i in range(2):
                        image = np.full((30, 40, 3), (10 + i, 60, 130), np.uint8)
                        cv2.imencode(".png", image)[1].tofile(str(folder / f"{i}.png"))
            model = keras.Sequential([
                keras.layers.Input((100, 100, 3)), keras.layers.Rescaling(1 / 255),
                keras.layers.GlobalAveragePooling2D(), keras.layers.Dense(7, activation="softmax")])
            source, output = root / "initial.keras", root / "fine_tuned.keras"
            model.save(source)
            original = source.read_bytes()
            args = ["--data-dir", str(data), "--initial-model", str(source),
                    "--output", str(output), "--report-dir", str(root / "reports"),
                    "--epochs", "1", "--batch-size", "7", "--unfreeze-layers", "1", "--evaluate-test"]
            self.assertEqual(training.main(args), 0)
            self.assertEqual(source.read_bytes(), original)
            best = keras.models.load_model(output, compile=False)
            self.assertEqual(best.output_shape[-1], 7)
            report = json.loads((root / "reports" / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "complete")
            self.assertGreaterEqual(report["best_validation"]["macro_f1"], report["baseline_validation"]["macro_f1"] - 1e-6)
            self.assertEqual(set(report["test"]["per_class"]), set(training.LABELS))
            with (root / "reports" / "metrics_per_class.csv").open(encoding="utf-8-sig") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 14)
            with self.assertRaises(SystemExit) as caught:
                training.main(args)
            self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
