"""Regression tests for cached YOLO inference diagnostics (no GPU/weights)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_cached import (
    build_report, detections, evaluate, iou, join_caches, matched, nms, partition,
)


def target(cls=1, size=0.2):
    return {"class": cls, "box": [0, 0, 10, 10], "area": size}


def prediction(cls=1, conf=0.7, box=None):
    return {"class": cls, "conf": conf, "box": box or [0, 0, 10, 10]}


def sample(name="plate__a.jpg", targets=None, full=None, crops=None, high=None):
    return {"image": name, "targets": targets if targets is not None else [target()],
            "full": full if full is not None else [prediction()],
            "crops": crops or [], "high": high or []}


class CachedAnalysisTests(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(iou([0, 0, 2, 2], [1, 1, 3, 3]), 1 / 7)
        self.assertEqual(iou([0, 0, 1, 1], [2, 2, 3, 3]), 0)

    def test_nms_preserves_different_class_boxes(self):
        self.assertEqual(len(nms([prediction(1), prediction(1, 0.6),
                                  prediction(8)], 0.5)), 2)

    def test_matching_one_prediction_cannot_claim_two_objects(self):
        self.assertEqual(len(matched([target(), target()], [prediction()])), 1)
        self.assertEqual(len(matched([target()], [prediction(8)])), 0)

    def test_full_frame_resolution_comparison(self):
        item = sample()
        self.assertEqual(evaluate([item], "full640")["overall"]["matched"], 1)
        self.assertEqual(evaluate([item], "full960")["overall"]["matched"], 0)

    def test_higher_crop_threshold_filters_unmatched_predictions(self):
        item = sample(crops=[prediction(8, 0.30)])
        self.assertEqual(evaluate([item], "640_plus_crops", 0.25, 0.5)
                         ["overall"]["unmatched_predictions"], 1)
        self.assertEqual(evaluate([item], "640_plus_crops", 0.35, 0.5)
                         ["overall"]["unmatched_predictions"], 0)

    def test_full_frame_confidence_sweep_changes_detections(self):
        item = sample(full=[prediction(1, 0.18)])
        self.assertEqual(len(detections(item, "full640", 0.15)), 1)
        self.assertEqual(len(detections(item, "full640", 0.25)), 0)

    def test_cache_checkpoint_mismatch_rejected(self):
        first = {"image": "x", "input_signature": [3, 1, "abc"],
                 "settings": {"weights_sha256_prefix": "a", "imgsz": 640, "device": "mps"},
                 "targets": [target()], "full": [], "crops": []}
        second = {"image": "x", "input_signature": [3, 1, "abc"],
                  "settings": {"weights_sha256_prefix": "b", "compare_imgsz": 960, "device": "mps"},
                  "full": []}
        with self.assertRaises(ValueError):
            join_caches({"x": first}, {"x": second})

    def test_development_holdout_partition(self):
        rows = [sample(f"plate__example_{i}.jpg") for i in range(70)]
        development, holdout = partition(rows)
        self.assertEqual(len(development) + len(holdout), len(rows))
        self.assertFalse({r["image"] for r in development} &
                         {r["image"] for r in holdout})
        report = build_report(rows)
        self.assertEqual(report["images"], 70)
        self.assertTrue(any(c["method"] == "full640" and c["crop_conf"] == 0.25
                            for c in report["candidates"]))


if __name__ == "__main__":
    unittest.main()
