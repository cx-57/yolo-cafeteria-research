"""Fast V2 integration tests: reviewed photos, leakage, source immutability and trials."""
import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train
import v2


class V2WorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "training"
        self.output = self.root / "training-v2"
        self.nhs = self.root / "nhs"
        self.add_source("train", "trash_detection__train__wrap.png", "9 .5 .5 .4 .4\n",
                        (240, 100, 50))
        self.add_source("val", "plate__val__small.png", "1 .5 .5 .4 .4\n",
                        (100, 150, 200))
        self.source_yaml = self.source / "data.yaml"
        self.source_yaml.write_text(yaml.safe_dump({
            "path": str(self.source),
            "train": str(self.source / "images/train"),
            "val": str(self.source / "images/val"),
            "names": train.class_names(),
        }))
        self.original_labels = {str(p): p.read_bytes()
                                for p in self.source.rglob("*.txt")}

    def add_source(self, split, filename, label, rgb):
        images = self.source / "images" / split
        labels = self.source / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (48, 36), rgb).save(images / filename)
        (labels / (Path(filename).stem + ".txt")).write_text(label)

    def add_nhs(self, split, session, n, full_classes=True):
        self.nhs.mkdir(exist_ok=True)
        image = self.nhs / f"photo_{n}.png"
        label = self.nhs / f"photo_{n}.txt"
        Image.new("RGB", (72, 45), (n * 11, 30, 100)).save(image)
        rows = [f"{i} .5 .5 .1 .1" for i in range(11)] if full_classes else []
        label.write_text("\n".join(rows) + ("\n" if rows else ""))
        return {"image": image.name, "label": label.name, "split": split,
                "session": session, "reviewer": "reviewer_a",
                "reviewed": "yes", "complete": "yes"}

    def manifest(self, rows):
        path = self.nhs / "manifest.csv"
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=[
                "image", "label", "split", "session", "reviewer", "reviewed", "complete"])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def args(self, **changes):
        base = dict(source=self.source, output=self.output, nhs_manifest=None,
                    label_additions=None, reviewed_overrides=None, provisional=True)
        base.update(changes)
        return argparse.Namespace(**base)

    def train_args(self, **changes):
        base = dict(data=self.output / "data.yaml", trial="control-n",
                    epochs=2, batch=2, device="cpu", dry_run=True,
                    allow_provisional=False)
        base.update(changes)
        return argparse.Namespace(**base)

    def test_provisional_uses_original_split_and_keeps_source_intact(self):
        data = v2.build(self.args())
        output = yaml.safe_load(data.read_text())
        self.assertEqual(len(list(Path(output["train"]).iterdir())), 1)
        self.assertEqual(len(list(Path(output["val"]).iterdir())), 1)
        self.assertFalse((self.output / "v2_build.json").read_text().find('"provisional": true') < 0)
        self.assertEqual(self.original_labels,
                         {str(p): p.read_bytes() for p in self.source.rglob("*.txt")})
        self.assertFalse((self.output / "images/test").exists())

    def test_strict_requires_complete_class_coverage(self):
        with self.assertRaisesRegex(ValueError, "coverage is incomplete"):
            v2.build(self.args(provisional=False))
        self.assertFalse(self.output.exists())

    def test_strict_moves_legacy_validation_into_train_copies(self):
        manifest = self.manifest([
            self.add_nhs("train", "day-a", 1),
            self.add_nhs("val", "day-b", 2),
            self.add_nhs("test", "day-c", 3),
        ])
        data = v2.build(self.args(provisional=False, nhs_manifest=manifest))
        spec = yaml.safe_load(data.read_text())
        training = {p.name for p in Path(spec["train"]).iterdir()}
        validation = {p.name for p in Path(spec["val"]).iterdir()}
        testing = {p.name for p in Path(spec["test"]).iterdir()}
        self.assertIn("plate__val__small.png", training)
        self.assertTrue(all(name.startswith("nhs__") for name in validation | testing))
        self.assertEqual((len(training), len(validation), len(testing)), (3, 1, 1))
        meta = json.loads((self.output / "v2_build.json").read_text())
        self.assertTrue(meta["strict_nhs_only_evaluation"])
        self.assertFalse(meta["provisional"])
        self.assertFalse(any(meta["final_audit"]["missing_classes"].values()))
        self.assertEqual(self.original_labels,
                         {str(p): p.read_bytes() for p in self.source.rglob("*.txt")})

    def test_nhs_session_leakage_rejected(self):
        manifest = self.manifest([
            self.add_nhs("val", "day-a", 1),
            self.add_nhs("test", "day-a", 2),
        ])
        with self.assertRaisesRegex(ValueError, "session crosses splits"):
            v2.build(self.args(nhs_manifest=manifest))
        self.assertFalse(self.output.exists())

    def test_exact_nhs_photo_repeated_rejected(self):
        item = self.add_nhs("train", "day-a", 1)
        other = {**item, "split": "val", "session": "day-b"}
        manifest = self.manifest([item, other])
        with self.assertRaises(ValueError):
            v2.build(self.args(nhs_manifest=manifest))
        self.assertFalse(self.output.exists())

    def test_v1_roboflow_augmented_export_cross_split_rejected(self):
        self.add_source("val",
                        "trash_detection__valid__wrap.rf.123.png",
                        "9 .5 .5 .4 .4\n", (220, 70, 40))
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            v2.build(self.args())
        self.assertFalse(self.output.exists())

    def test_verified_override_changes_copy_only(self):
        original = self.source / "labels/train/trash_detection__train__wrap.txt"
        corrections = self.root / "overrides.json"
        corrections.write_text(json.dumps([{
            "source": "trash_detection", "image": "train__wrap.png",
            "rows": ["8 .5 .5 .4 .4"], "complete": "yes",
            "reviewer": "reviewer_a", "reason": "manual wrapper visual verification",
            "expected_label_sha256": v2.digest(original),
        }]))
        v2.build(self.args(reviewed_overrides=corrections))
        self.assertEqual(original.read_text(), "9 .5 .5 .4 .4\n")
        corrected = self.output / "labels/train/trash_detection__train__wrap.txt"
        self.assertEqual(corrected.read_text(), "8 .5 .5 .4 .4\n")

    def test_unverified_override_and_stale_sha_rejected(self):
        path = self.root / "bad.json"
        path.write_text(json.dumps([{
            "source": "trash_detection", "image": "train__wrap.png",
            "rows": ["8 .5 .5 .4 .4"], "complete": "yes",
            "reviewer": "reviewer_a", "reason": "relabel",
            "expected_label_sha256": "stale",
        }]))
        with self.assertRaisesRegex(ValueError, "source label changed"):
            v2.build(self.args(reviewed_overrides=path))
        self.assertFalse(self.output.exists())

    def test_provisional_training_gated_and_hashes_checked(self):
        v2.build(self.args())
        with self.assertRaisesRegex(ValueError, "provisional"):
            v2.train(self.train_args())
        v2.train(self.train_args(allow_provisional=True))
        (self.output / "labels/train/trash_detection__train__wrap.txt").write_text(
            "8 .5 .5 .4 .4\n")
        with self.assertRaisesRegex(ValueError, "content changed"):
            v2.train(self.train_args(allow_provisional=True))

    def test_trial_commands_differ_only_in_intended_options(self):
        v2.build(self.args())
        runs = []
        for trial in v2.TRIALS:
            with patch.object(v2.subprocess, "run", side_effect=lambda cmd, **kw: runs.append(cmd)):
                v2.train(self.train_args(trial=trial, allow_provisional=True,
                                         dry_run=False))
        def flags(cmd):
            return {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1)
                    if cmd[i].startswith("--") and not cmd[i + 1].startswith("--")}
        a, b, c = map(flags, runs)
        self.assertEqual(a["--model"], b["--model"])
        self.assertEqual(b["--cls-pw"], "0.25")
        self.assertEqual(c["--model"], "yolo11s.pt")
        self.assertEqual(a["--imgsz"], b["--imgsz"])
        self.assertEqual(a["--imgsz"], c["--imgsz"])
        self.assertEqual(a["--seed"], b["--seed"])


if __name__ == "__main__":
    unittest.main()
