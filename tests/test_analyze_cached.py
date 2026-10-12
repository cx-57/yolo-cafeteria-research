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


class DetailedMatchingTests(unittest.TestCase):
    def test_max_matching_fixes_overlapping_plate_greedy_error(self):
        from analyze_cached import assignments
        first = {"class": 1, "box": [0, 0, 10, 10], "area": 0.02}
        second = {"class": 1, "box": [6, 0, 16, 10], "area": 0.02}
        high = prediction(1, 0.9, [2.9, 0, 12.9, 10])
        low = prediction(1, 0.7, [0, 0, 10, 10])
        self.assertEqual(len(assignments([first, second], [high, low])), 2)

    def test_class_counts_nonnegative_and_size_report(self):
        result = evaluate([sample()], "full640")
        plate = result["classes"]["plate"]
        self.assertEqual((plate["predictions"], plate["matched"],
                          plate["unmatched_predictions"]), (1, 1, 0))
        self.assertEqual(result["sizes"]["large_16pct_plus"]["matched"], 1)
        self.assertNotIn("unmatched_predictions",
                         result["sizes"]["large_16pct_plus"])

    def test_roboflow_exports_stay_in_same_partition(self):
        from analyze_cached import source_group
        a = "trash_detection__train__000055_jpg.rf.aaaa.jpg"
        b = "trash_detection__valid__000055_jpg.rf.bbbb.jpg"
        self.assertEqual(source_group(a), source_group(b))
        one = sample(name=a)
        two = sample(name=b)
        # Only two images in partition cannot guarantee nonempty sides, but
        # each image deterministically hashes to the same side.
        self.assertEqual(
            __import__("hashlib").sha256(("42:" + source_group(a)).encode()).digest(),
            __import__("hashlib").sha256(("42:" + source_group(b)).encode()).digest())

    def test_wrong_class_overlap_flags_review_candidate(self):
        item = sample(full=[prediction(8)])
        candidates = evaluate([item], "full640")["unmatched_review"]
        self.assertEqual(candidates[0]["review_hint"], "potential_class_confusion")


    def test_render_overlay_writes_review_file_without_modifying_source(self):
        import tempfile
        from PIL import Image
        from analyze_cached import render_review_overlays
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "plate__test.jpg"
            Image.new("RGB", (100, 80), "white").save(image)
            original = image.read_bytes()
            item = sample(name=image.name, crops=[prediction(8)])
            count = render_review_overlays([item], root, root / "review", 1)
            self.assertEqual(count, 1)
            self.assertEqual(image.read_bytes(), original)
            self.assertEqual(len(list((root / "review").glob("*.jpg"))), 1)



if __name__ == "__main__":
    unittest.main()
