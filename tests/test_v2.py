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
                    allow_provisional=False, tag=None)
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

    def test_strict_nhs_validation_does_not_require_test_set(self):
        manifest = self.manifest([
            self.add_nhs("train", "day-a", 1),
            self.add_nhs("val", "day-b", 2),
        ])
        path = v2.build(self.args(provisional=False, nhs_manifest=manifest))
        spec = yaml.safe_load(path.read_text())
        self.assertNotIn("test", spec)
        self.assertEqual(len(list(Path(spec["val"]).iterdir())), 1)
        self.assertTrue(all(p.name.startswith("nhs__")
                            for p in Path(spec["val"]).iterdir()))
        metadata = json.loads((self.output / "v2_build.json").read_text())
        self.assertFalse(metadata["provisional"])
        self.assertFalse(metadata["independent_test_available"])
        self.assertIn("No independent test", metadata["evaluation_note"])

    def test_strict_test_when_supplied_needs_class_coverage(self):
        manifest = self.manifest([
            self.add_nhs("train", "day-a", 1),
            self.add_nhs("val", "day-b", 2),
            self.add_nhs("test", "day-c", 3, full_classes=False),
        ])
        with self.assertRaisesRegex(ValueError, "coverage is incomplete"):
            v2.build(self.args(provisional=False, nhs_manifest=manifest))
        self.assertFalse(self.output.exists())

    def test_unverified_candidate_train_only_and_explicit(self):
        item = self.add_nhs("train", "lvis-source-1", 1)
        item["reviewed"] = "no"
        item["complete"] = "no"
        manifest = self.manifest([item])
        result = v2.build(self.args(candidate_manifest=manifest))
        config = yaml.safe_load(result.read_text())
        self.assertEqual(
            len([p for p in Path(config["train"]).iterdir()
                 if p.name.startswith("candidate__")]), 1)
        self.assertEqual(
            len([p for p in Path(config["val"]).iterdir()
                 if p.name.startswith("candidate__")]), 0)
        report = json.loads((self.output / "v2_build.json").read_text())
        self.assertEqual(report["unverified_candidate_images"], 1)
        self.assertTrue(report["provisional"])

    def test_unverified_candidates_refused_in_strict_or_validation(self):
        item = self.add_nhs("train", "lvis-source-1", 1)
        item["reviewed"] = "no"
        item["complete"] = "no"
        manifest = self.manifest([item])
        with self.assertRaisesRegex(ValueError, "only with --provisional"):
            v2.build(self.args(candidate_manifest=manifest, provisional=False))
        self.assertFalse(self.output.exists())
        item["split"] = "val"
        manifest = self.manifest([item])
        with self.assertRaisesRegex(ValueError, "stay in training"):
            v2.build(self.args(candidate_manifest=manifest))
        self.assertFalse(self.output.exists())

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

    def test_roboflow_cross_split_group_is_repaired_only_in_v2(self):
        name = "trash_detection__valid__wrap.rf.123.png"
        self.add_source("val", name, "9 .5 .5 .4 .4\n", (220, 70, 40))
        originals = {str(p): p.read_bytes()
                     for p in self.source.rglob("*") if p.is_file()}
        result = v2.build(self.args())
        spec = yaml.safe_load(result.read_text())
        training = {p.name for p in Path(spec["train"]).iterdir()}
        validation = {p.name for p in Path(spec["val"]).iterdir()}
        self.assertIn(name, training)
        self.assertNotIn(name, validation)
        report = json.loads((self.output / "v2_build.json").read_text())
        self.assertEqual(report["legacy_validation_images_moved"], 1)
        self.assertEqual(len(report["legacy_split_repairs"]), 1)
        self.assertEqual(originals, {str(p): p.read_bytes()
                                    for p in self.source.rglob("*") if p.is_file()})

    def test_transitive_duplicate_hash_repair_moves_related_images(self):
        self.add_source("val", "trash_detection__valid__wrap.rf.123.png",
                        "9 .5 .5 .4 .4\n", (220, 70, 40))
        self.add_source("val", "extra__valid__unrelated.png",
                        "8 .5 .5 .4 .4\n", (220, 70, 40))
        result = v2.build(self.args())
        spec = yaml.safe_load(result.read_text())
        names = {p.name for p in Path(spec["train"]).iterdir()}
        self.assertIn("trash_detection__valid__wrap.rf.123.png", names)
        self.assertIn("extra__valid__unrelated.png", names)
        report = json.loads((self.output / "v2_build.json").read_text())
        self.assertEqual(report["legacy_validation_images_moved"], 2)
        self.assertEqual(len(report["legacy_split_repairs"]), 1)

    def test_repair_cannot_empty_validation(self):
        original = self.source / "images/train/trash_detection__train__wrap.png"
        val = self.source / "images/val/plate__val__small.png"
        val.write_bytes(original.read_bytes())
        with self.assertRaisesRegex(ValueError, "All validation images"):
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

    def test_review_queue_does_not_confuse_food_only_with_full_review(self):
        full = self.root / "verified_full.json"
        food = self.root / "verified_food_only.json"
        full.write_text(json.dumps([{"source": "trash_detection", "image": "train__wrap.png"}]))
        food.write_text(json.dumps([{"source": "plate", "image": "val__small.png"}]))
        output = self.root / "generated_review_queue.csv"
        counts = v2.make_review_queue(self.source_yaml, output, full, food)
        self.assertEqual(counts, {"full_image_verified": 1,
                                  "food_only_verified": 1, "not_verified": 0})
        with output.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["review_status"] for row in rows},
                         {"full_image_verified", "food_only_verified"})
        food_only = next(row for row in rows
                         if row["review_status"] == "food_only_verified")
        self.assertEqual(food_only["requires_all_11_class_review"], "yes")
        self.assertEqual(len(food_only["label_sha256"]), 64)

    def test_full_resolution_unchanged_labels_count_but_flags_do_not(self):
        review_root = self.root / "data/audits/v2_review_2026-10-12"
        review_root.mkdir(parents=True)
        (review_root / "image_review.csv").write_text(
            "source,image,review_scope,decision\n"
            "trash_detection,train__wrap.png,full_resolution,checked_existing\n"
            "plate,val__small.png,full_resolution,needs_followup\n"
        )
        empty = self.root / "empty_reviews.json"
        empty.write_text("[]")
        with patch.object(v2, "PROJECT_ROOT", self.root):
            counts = v2.make_review_queue(
                self.source_yaml, self.root / "review_log_queue.csv",
                overrides_file=empty, food_file=empty)
        self.assertEqual(counts, {"full_image_verified": 1,
                                  "food_only_verified": 0, "not_verified": 1})

    def test_codex_review_preset_applies_both_approved_sources(self):
        audit = self.root / "data/audits/v2_review_2026-10-12"
        audit.mkdir(parents=True)
        (audit / "lvis_food_approved.json").write_text(json.dumps([{
            "source": "plate", "image": "val__small.png",
            "rows": ["0 .2 .2 .1 .1"], "provenance": {"reviewed": "yes"}
        }]))
        original = self.source / "labels/train/trash_detection__train__wrap.txt"
        (audit / "reviewed_overrides.json").write_text(json.dumps([{
            "source": "trash_detection", "image": "train__wrap.png",
            "rows": ["8 .5 .5 .4 .4"], "complete": "yes",
            "reviewer": "reviewer", "reason": "Manually verified wrapper",
            "expected_label_sha256": v2.digest(original)
        }]))
        with patch.object(v2, "PROJECT_ROOT", self.root):
            v2.build(self.args(use_codex_review=True))
        self.assertEqual(
            (self.output / "labels/train/trash_detection__train__wrap.txt").read_text(),
            "8 .5 .5 .4 .4\n")
        self.assertIn("0 .2 .2 .1 .1",
                      (self.output / "labels/val/plate__val__small.txt").read_text())
        report = json.loads((self.output / "v2_build.json").read_text())
        self.assertTrue(report["reviewed_audit_applied"])
        self.assertEqual(report["legacy_validation_images_moved"], 0)
        self.assertEqual(original.read_text(), "9 .5 .5 .4 .4\n")

    def test_codex_review_preset_rejects_ambiguous_explicit_files(self):
        with self.assertRaisesRegex(ValueError, "do not combine"):
            v2.build(self.args(use_codex_review=True,
                               reviewed_overrides=self.root / "another.json"))
        self.assertFalse(self.output.exists())

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
        runs = [v2.train(self.train_args(trial=trial, allow_provisional=True))
                for trial in v2.TRIALS]
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

    def test_reviewed_public_validation_is_not_reported_as_nhs(self):
        manifest = self.manifest([
            self.add_nhs("train", "public-photo-a", 1),
            self.add_nhs("val", "public-photo-b", 2),
        ])
        path = v2.build(self.args(provisional=False, reviewed_manifest=manifest))
        meta = json.loads((self.output / "v2_build.json").read_text())
        self.assertFalse(meta["strict_nhs_only_evaluation"])
        self.assertEqual(meta["evaluation_domain"], "reviewed_public")
        self.assertTrue(all(p.name.startswith("reviewed__")
                            for p in Path(yaml.safe_load(path.read_text())["val"]).iterdir()))

    def test_full_comparison_disables_early_stopping(self):
        v2.build(self.args())
        cmd = v2.train(self.train_args(epochs=100, allow_provisional=True))
        self.assertIn("--patience", cmd)
        self.assertEqual(cmd[cmd.index("--patience") + 1], "0")

    def test_ranking_rejects_different_training_schedules(self):
        entries = [dict(trial="control-n", epochs=100, batch=4, seed=42, imgsz=640,
                        dataset_sha256="frozen"),
                   dict(trial="control-s", epochs=80, batch=4, seed=42, imgsz=640,
                        dataset_sha256="frozen")]
        with self.assertRaisesRegex(ValueError, "schedule"):
            v2.check_trial_schedules(entries)


if __name__ == "__main__":
    unittest.main()
