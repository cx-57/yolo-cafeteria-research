"""Offline regression tests for visual YOLO inference review (fake model, no weights)."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import inspect_predictions as review


class InspectionTests(unittest.TestCase):
    def test_same_class_iou_matching_and_unmatched_objects(self):
        gt = [(1, (10, 10, 60, 60), None), (1, (80, 80, 100, 100), None)]
        pred = [(1, (12, 12, 60, 60), .8), (0, (80, 80, 100, 100), .9)]
        matched_gt, matched_pred = review.match(gt, pred, 0.5)
        self.assertEqual(matched_gt, {0})
        self.assertEqual(matched_pred, {0})
        self.assertEqual(review.box_iou((0,0,1,1),(2,2,3,3)), 0)

    def test_overlapping_objects_use_maximum_number_of_distinct_matches(self):
        # Greedy IoU matching picks A->1 and loses B. Augmenting paths
        # should reassign A->2 so that B->1 is also correctly counted.
        targets = [(1, (0,), None), (1, (1,), None)]
        predictions = [(1, (0,), .8), (1, (1,), .7)]
        weights = {(0, 0): .95, (0, 1): .8,
                   (1, 0): .8, (1, 1): .1}
        with patch.object(review, "box_iou",
                          side_effect=lambda a, b: weights[(a[0], b[0])]):
            matched_gt, matched_pred = review.match(targets, predictions, 0.5)
        self.assertEqual(matched_gt, {0, 1})
        self.assertEqual(matched_pred, {0, 1})

    def test_wrong_class_overlap_is_explicitly_flagged_not_reclassified(self):
        targets = [(8, (0, 0, 100, 100), None)]
        preds = [(9, (2, 2, 98, 98), .73),
                 (8, (200, 200, 250, 250), .61)]
        matched_gt, matched_pred = review.match(targets, preds, .5)
        self.assertEqual((matched_gt, matched_pred), (set(), set()))
        conflicts = review.wrong_class_candidates(
            targets, preds, matched_gt, matched_pred, .5)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0][:2], (0, 0))
        self.assertGreater(conflicts[0][2], .9)
        self.assertEqual(targets[0][0], 8)  # Never rewrite ground truth.

    def test_size_bins_and_stratified_review_keep_occluded_candidates(self):
        self.assertEqual(review.size_band(.009), "tiny_under_1pct")
        self.assertEqual(review.size_band(.01), "small_1_to_4pct")
        self.assertEqual(review.size_band(.04), "medium_4_to_16pct")
        self.assertEqual(review.size_band(.16), "large_16pct_plus")
        self.assertAlmostEqual(review.area_fraction((0, 0, 10, 10), 100, 100), .01)
        samples = [(Path(f"plate_{n}.jpg"),
                    [(1, (0, 0, (n+1)*2, (n+1)*2), None)], (100, 100))
                   for n in range(8)]
        sampled = review.sample_across_sizes(samples[::-1], 1, 3)
        self.assertEqual([item[0].stem for item in sampled],
                         ["plate_0", "plate_4", "plate_7"])
        self.assertEqual(len(review.sample_across_sizes(samples, 1, 100)), 8)

    def test_real_image_overlay_files_and_zero_validation_support(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = root/"images/val"
            labels = root/"labels/val"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            Image.new("RGB", (160, 120), "white").save(images/"plate__partly_hidden.jpg")
            (labels/"plate__partly_hidden.txt").write_text("1 .5 .5 .7 .6\n")
            yml = root/"data.yaml"
            yml.write_text(yaml.safe_dump({"path": str(root), "train": "images/train",
                                           "val": "images/val", "names": list(review.NAMES)}))
            box = SimpleNamespace(cls=SimpleNamespace(item=lambda:1),
                                  xyxy=[[15.0, 25.0, 110.0, 85.0]],
                                  conf=SimpleNamespace(item=lambda:.8))
            # Mimic ultralytics box indexing: box.xyxy[0] gives iterable scalars.
            class FakeYOLO:
                def __init__(self, path):
                    self.names = dict(enumerate(review.NAMES))
                def predict(self, *args, **kwargs):
                    return [SimpleNamespace(boxes=[box])]
            with patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=FakeYOLO)}):
                review.inspect(yml, "fake.pt", root/"output", "val",
                               ["plate", "napkin"], 2, .1, .5, None)
            report = json.loads((root/"output/summary.json").read_text())
            self.assertEqual(report["images_reviewed"], 1)
            self.assertEqual(report["eligible_images_by_class"], {"plate":1, "napkin":0})
            self.assertEqual(report["reviewed"][0]["pred"], 1)
            self.assertEqual(report["per_class"]["plate"]["target_count"], 1)
            self.assertEqual(report["per_class"]["plate"]["matched"], 1)
            self.assertEqual(report["by_object_area"]["plate"]["large_16pct_plus"]["matched"], 1)
            self.assertTrue((root/"output/review_queue.csv").is_file())
            self.assertTrue((root/"output/wrong_class_candidates.csv").is_file())
            with Image.open(root/"output/plate__partly_hidden.jpg") as im:
                self.assertEqual(im.size, (320, 120))
            with patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=FakeYOLO)}):
                review.inspect(yml, "fake.pt", root/"full-report", "val",
                               ["plate"], 1, .1, .5, None,
                               all_images=True, save_overlays=False)
            full = json.loads((root/"full-report/summary.json").read_text())
            self.assertEqual(full["selection"], "full_split")
            self.assertEqual(full["images_reviewed"], 1)
            self.assertFalse(full["overlays_saved"])
            self.assertFalse((root/"full-report/plate__partly_hidden.jpg").exists())
            self.assertTrue((root/"full-report/wrong_class_candidates.csv").exists())

    def test_bad_checkpoint_ontology_is_rejected_before_predictions(self):
        class FakeYOLO:
            def __init__(self, path):
                self.names = {0: "wrong"}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"data.yaml").write_text(yaml.safe_dump({"names":list(review.NAMES)}))
            with patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=FakeYOLO)}):
                with self.assertRaisesRegex(ValueError, "Checkpoint"):
                    review.inspect(root/"data.yaml", "fake.pt", root/"output", "val",
                                   ["plate"], 1, .1, .5, None)


if __name__ == "__main__":
    unittest.main()
