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
            with Image.open(root/"output/plate__partly_hidden.jpg") as im:
                self.assertEqual(im.size, (320, 120))

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
