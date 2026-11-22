"""Kiểm thử các cảnh báo chất lượng ảnh không cần nạp mô hình TensorFlow."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))

from image_quality import analyze_face_quality, analyze_frame_quality, low_confidence_warning


class ImageQualityTests(unittest.TestCase):
    def test_dark_blurred_frame_is_rejected(self):
        report = analyze_frame_quality(np.full((240, 320, 3), 20, np.uint8))
        codes = {warning["code"] for warning in report["warnings"]}
        self.assertFalse(report["ok"])
        self.assertIn("too_dark", codes)
        self.assertIn("blurred", codes)

    def test_face_position_pose_and_confidence_warnings(self):
        rng = np.random.default_rng(7)
        frame = rng.integers(70, 190, (480, 640, 3), dtype=np.uint8)
        face = {
            "box": (0, 160, 50, 50), "angle": 20,
            "pose": {"roll": 20, "yaw": 24, "pitch": 19},
        }
        report = analyze_face_quality(frame, face)
        codes = {warning["code"] for warning in report["warnings"]}
        self.assertTrue({"too_far", "face_clipped", "head_tilt", "head_turn", "head_down"} <= codes)
        self.assertEqual(low_confidence_warning(0.30)["code"], "low_confidence")
        self.assertIsNone(low_confidence_warning(0.80))


if __name__ == "__main__":
    unittest.main()
