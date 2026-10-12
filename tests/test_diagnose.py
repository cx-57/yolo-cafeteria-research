"""Regression checks for support reporting and fixed-validation source ablations."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]


class DiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset = self.root / "training"
        self.output = self.root / "report"
        self.dataset.mkdir()
        (self.dataset / "data.yaml").write_text(yaml.safe_dump({
            "path": str(self.dataset), "train": "images/train", "val": "images/val",
            "names": ["food", "plate", "paper_boat"],
        }))

    def add_image(self, split, name, rows):
        images = self.dataset / "images" / split
        labels = self.dataset / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        image = images / (name + ".png")
        Image.new("RGB", (200, 100), "white").save(image)
        (labels / (name + ".txt")).write_text(rows)
        return image

    def run_diagnosis(self, *args):
        return subprocess.run([
            sys.executable, str(ROOT / "scripts/diagnose.py"),
            "--dataset", str(self.dataset), "--output", str(self.output), *args,
        ], text=True, capture_output=True)

    def test_support_distinguishes_boxes_images_and_missing_classes(self):
        self.add_image("train", "school_lunch__a", "0 .5 .5 .4 .4\n0 .2 .2 .09 .09\n")
        self.add_image("train", "plate__b", "1 .5 .5 .8 .8\n")
        self.add_image("val", "school_lunch__c", "0 .5 .5 .4 .4\n")
        result = self.run_diagnosis()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((self.output / "diagnosis.json").read_text())
        self.assertEqual(report["splits"]["train"]["classes"]["food"]["instances"], 2)
        self.assertEqual(report["splits"]["train"]["classes"]["food"]["images"], 1)
        self.assertEqual(report["missing_train_classes"], ["paper_boat"])
        self.assertEqual(report["missing_val_classes"], ["plate", "paper_boat"])
        self.assertEqual(report["splits"]["train"]["classes"]["food"]["short_side_under_32px"], 1)

    def test_ablation_filters_train_without_moving_validation_or_editing_labels(self):
        food = self.add_image("train", "school_lunch__a", "0 .5 .5 .4 .4\n")
        self.add_image("train", "plate__b", "1 .5 .5 .8 .8\n")
        val = self.add_image("val", "plate__c", "1 .5 .5 .8 .8\n")
        before = {p: p.read_bytes() for p in self.dataset.rglob("*.txt")}
        result = self.run_diagnosis("--exclude-training-source", "plate")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = yaml.safe_load((self.output / "ablation/data.yaml").read_text())
        self.assertEqual(Path(data["train"]).read_text().splitlines(), [str(food.resolve())])
        self.assertEqual(Path(data["val"]).resolve(), val.parent.resolve())
        self.assertEqual(before, {p: p.read_bytes() for p in self.dataset.rglob("*.txt")})
        self.assertEqual(data["names"], ["food", "plate", "paper_boat"])

    def test_run_summary_reports_missing_epochs_and_best_metrics_separately(self):
        self.add_image("train", "school_lunch__a", "0 .5 .5 .4 .4\n")
        self.add_image("val", "school_lunch__c", "0 .5 .5 .4 .4\n")
        run = self.root / "run"
        run.mkdir()
        (run / "args.yaml").write_text("epochs: 5\nclose_mosaic: 1\n")
        fields = ["epoch", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
        with (run / "results.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            writer.writerows([[1, .74, .48], [3, .73, .50]])
        result = self.run_diagnosis("--run", str(run))
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads((self.output / "diagnosis.json").read_text())["run"]
        self.assertEqual(summary["missing_epochs"], [2])
        self.assertEqual(summary["best"]["metrics/mAP50(B)"], {"epoch": 1, "value": .74})
        self.assertEqual(summary["best"]["metrics/mAP50-95(B)"], {"epoch": 3, "value": .50})
        self.assertFalse(summary["scheduled_mosaic_close_reached"])

    def test_unknown_exclusion_fails_instead_of_creating_an_unchanged_experiment(self):
        self.add_image("train", "school_lunch__a", "0 .5 .5 .4 .4\n")
        self.add_image("val", "school_lunch__c", "0 .5 .5 .4 .4\n")
        result = self.run_diagnosis("--exclude-training-source", "typo")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unknown training source", result.stderr)

    def test_metadata_mode_avoids_label_reads_and_does_not_invent_box_counts(self):
        self.add_image("train", "school_lunch__a", "unavailable annotation\n")
        self.add_image("val", "school_lunch__c", "unavailable annotation\n")
        (self.dataset / "split_summary.json").write_text(json.dumps({
            "train_views": 1, "train_originals": 1, "val_originals": 1,
            "augment_factor": 1,
            "train_class_images_original": {"food": 1, "plate": 0, "paper_boat": 0},
            "val_class_images_original": {"food": 1, "plate": 0, "paper_boat": 0},
        }))
        result = self.run_diagnosis("--metadata-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((self.output / "diagnosis.json").read_text())
        self.assertEqual(report["missing_train_classes"], ["plate", "paper_boat"])
        food = report["splits"]["train"]["classes"]["food"]
        self.assertEqual(food["images"], 1)
        self.assertIsNone(food["instances"])
        self.assertIsNone(food["short_side_under_32px"])
        self.assertEqual(report["support_basis"], "saved_split_summary_not_fresh_label_scan")


if __name__ == "__main__":
    unittest.main()
