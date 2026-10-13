"""Frozen-split experiments retain every image and ground-truth annotation."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import experiment
import train


class FrozenExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "data/training"
        self.output = self.root / "data/training-revised"
        self.source.mkdir(parents=True)
        self.source_yaml = self.source / "data.yaml"
        self.source_yaml.write_text(yaml.safe_dump({
            "path": str(self.source), "train": "images/train", "val": "images/val",
            "names": train.class_names()}))
        root_patch = patch.object(experiment, "PROJECT_ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        for i in range(5):
            self.add_image("train", f"meal__food_{i}", "0 .5 .5 .4 .4\n")
        self.add_image("train", "meal__plate", "0 .5 .5 .4 .4\n1 .5 .5 .8 .8\n")
        self.add_image("train", "meal__wrapper", "0 .5 .5 .4 .4\n8 .2 .2 .2 .2\n")
        self.add_image("train", "meal__bottle", "0 .5 .5 .4 .4\n9 .8 .8 .2 .2\n")
        self.add_image("val", "meal__occluded", "0 .5 .5 .4 .4\n1 .5 .5 .8 .8\n")
        self.add_image("val", "meal__wrapper_val", "8 .2 .2 .2 .2\n")

    def add_image(self, split, stem, labels):
        images = self.source / "images" / split
        targets = self.source / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        targets.mkdir(parents=True, exist_ok=True)
        path = images / (stem + ".png")
        Image.new("RGB", (57, 31), (80, 20, len(stem))).save(path)
        (targets / (stem + ".txt")).write_text(labels)
        return path

    def additions(self, records):
        path = self.root / "label_additions.json"
        path.write_text(json.dumps(records))
        return path

    def summary(self):
        return json.loads((self.output / "experiment_summary.json").read_text())

    def source_bytes(self):
        return {str(path.relative_to(self.source)): path.read_bytes()
                for path in self.source.rglob("*") if path.is_file()}

    def test_rare_class_training_views_repeat_boundedly_and_validation_has_one_view(self):
        original = self.source_bytes()
        data_file = experiment.prepare_experiment(
            self.source, self.output, balance_classes=["plate", "wrapper", "bottle"], max_repeat=3)
        self.assertEqual(self.source_bytes(), original)
        train_images = sorted((self.output / "images/train").glob("*.png"))
        val_images = sorted((self.output / "images/val").glob("*.png"))
        self.assertEqual(len(train_images), 14)  # 5 common + 3 rare originals with 3 views each.
        self.assertEqual(len(val_images), 2)
        for stem in ("meal__plate", "meal__wrapper", "meal__bottle"):
            self.assertEqual(len(list((self.output / "images/train").glob(stem + "*.png"))), 3)
            self.assertTrue((self.output / "images/train" / (stem + "__balanced01.png")).exists())
        self.assertFalse(any("__balanced" in image.name for image in val_images))
        for split in ("train", "val"):
            for source in (self.source / "images" / split).iterdir():
                output = self.output / "images" / split / source.name
                self.assertEqual(output.read_bytes(), source.read_bytes())
                self.assertEqual(output.stat().st_ino, source.stat().st_ino)
                with Image.open(output) as opened:
                    self.assertEqual(opened.size, (57, 31))
        data = yaml.safe_load(data_file.read_text())
        self.assertTrue(Path(data["train"]).is_absolute())
        self.assertTrue(Path(data["val"]).is_absolute())
        report = self.summary()
        self.assertEqual(report["counts"], {"train_originals": 8, "train_views": 14,
                                            "val_originals": 2, "val_views": 2})
        self.assertEqual(report["original_support"]["train"]["plate"]["images"], 1)
        self.assertEqual(report["view_support"]["train"]["plate"]["images"], 3)
        self.assertEqual(report["source_yaml_sha256"], hashlib.sha256(self.source_yaml.read_bytes()).hexdigest())
        self.assertEqual(report["source_image_paths"]["val"], sorted(
            str(path.resolve()) for path in (self.source / "images/val").iterdir()))

    def test_focus_crops_only_train_and_matches_repeat_control(self):
        # Crop around a small wrapper while preserving both the frozen
        # validation files and the original source dataset byte-for-byte.
        self.add_image("train", "meal__tiny_wrapper",
                       "8 .3 .3 .08 .08\n0 .9 .9 .05 .05\n")
        before = self.source_bytes()
        control = self.root / "data/training-repeat-control"
        experiment.prepare_experiment(
            self.source, control, balance_classes=["wrapper"], max_repeat=2)
        experiment.prepare_experiment(
            self.source, self.output, balance_classes=["wrapper"],
            max_repeat=2, crop_repeats=True)
        self.assertEqual(self.source_bytes(), before)
        control_report = json.loads((control/"experiment_summary.json").read_text())
        cropped_report = self.summary()
        self.assertEqual(control_report["counts"], cropped_report["counts"])
        self.assertGreater(cropped_report["focus_crops"]["cropped_train_views"], 0)
        self.assertLess(
            cropped_report["view_support"]["train"]["food"]["instances"],
            control_report["view_support"]["train"]["food"]["instances"])
        for img in (self.source/"images/val").iterdir():
            for root in (control, self.output):
                self.assertEqual((root/"images/val"/img.name).read_bytes(), img.read_bytes())
                self.assertEqual((root/"labels/val"/(img.stem+".txt")).read_bytes(),
                                 (self.source/"labels/val"/(img.stem+".txt")).read_bytes())
        original = self.output/"images/train/meal__tiny_wrapper.png"
        cropped = self.output/"images/train/meal__tiny_wrapper__balanced01.png"
        self.assertEqual(original.read_bytes(),
                         (self.source/"images/train/meal__tiny_wrapper.png").read_bytes())
        with Image.open(cropped) as picture:
            self.assertLess(picture.width, 57)
            self.assertLess(picture.height, 31)
        rows = (self.output/"labels/train/meal__tiny_wrapper__balanced01.txt").read_text().splitlines()
        self.assertIn(8, {int(row.split()[0]) for row in rows})
        experiment.validate_rows(rows, cropped, train.class_names())
        width = float(next(row.split()[3] for row in rows if row.startswith("8 ")))
        self.assertGreater(width, 0.08)

    def test_partial_objects_in_crop_remain_labeled_and_boundary_valid(self):
        picture = self.add_image("train", "meal__partial",
                                 "8 .5 .5 .05 .05\n1 .73 .5 .20 .20\n")
        result = experiment.focus_crop(picture, [
            "8 .5 .5 .05 .05", "1 .73 .5 .20 .20"],
            train.class_names(), ["wrapper"], view=1)
        self.assertIsNotNone(result)
        _, rows = result
        self.assertEqual({int(row.split()[0]) for row in rows}, {1, 8})
        experiment.validate_rows(rows, "crop", train.class_names())

    def test_crop_requires_selected_weak_classes_and_repeat_budget(self):
        for choices, repeats in (([], 2), (["wrapper"], 1)):
            with self.subTest(choices=choices, repeats=repeats):
                with self.assertRaises(ValueError):
                    experiment.prepare_experiment(self.source, self.output,
                        balance_classes=choices, max_repeat=repeats, crop_repeats=True)
                self.assertFalse(self.output.exists())

    def test_a_mixed_rare_class_image_uses_the_maximum_repeat_not_a_sum(self):
        self.add_image("train", "meal__mixed", "1 .5 .5 .8 .8\n8 .2 .2 .2 .2\n9 .8 .8 .2 .2\n")
        experiment.prepare_experiment(self.source, self.output,
                                      balance_classes=["plate", "wrapper", "bottle"], max_repeat=3)
        self.assertEqual(len(list((self.output / "images/train").glob("meal__mixed*.png"))), 3)
        self.assertEqual(self.summary()["counts"]["train_views"], 17)

    def test_additions_append_to_both_splits_deduplicate_same_class_and_preserve_occluded_plate(self):
        before = self.source_bytes()
        path = self.additions([
            {"source": "meal", "image": "food_0.png", "rows": ["1 .1 .1 .15 .15"],
             "provenance": {"review": "visible plate"}},
            {"source": "meal", "image": "occluded.png",
             "rows": ["1 .505 .5 .8 .8", "9 .5 .5 .8 .8", "9 .5 .5 .8 .8"],
             "provenance": "reviewed additional object"}])
        experiment.prepare_experiment(self.source, self.output, label_additions=path)
        self.assertEqual(self.source_bytes(), before)
        train_label = (self.output / "labels/train/meal__food_0.txt").read_text()
        self.assertEqual(train_label, "0 .5 .5 .4 .4\n1 .1 .1 .15 .15\n")
        val_label = (self.output / "labels/val/meal__occluded.txt").read_text()
        self.assertEqual(val_label, "0 .5 .5 .4 .4\n1 .5 .5 .8 .8\n9 .5 .5 .8 .8\n")
        report = self.summary()
        self.assertEqual(report["additions"]["accepted"], 2)
        self.assertEqual(report["additions"]["duplicates_skipped"], 2)
        self.assertEqual(report["additions"]["by_split"], {"train": 1, "val": 1})
        self.assertEqual(report["original_support"]["val"]["bottle"]["instances"], 0)
        self.assertEqual(report["corrected_support"]["val"]["bottle"]["instances"], 1)
        self.assertEqual(report["evaluation_cohorts"]["val"]["plate"],
                         [str((self.source / "images/val/meal__occluded.png").resolve())])

    def test_current_ids_are_not_remapped_even_for_historical_source_names(self):
        self.add_image("train", "macworx__box", "4 .5 .5 .5 .5\n")
        additions = self.additions([{"source": "macworx", "image": "box.png",
                                    "rows": ["5 .2 .2 .2 .2"], "provenance": "review"}])
        experiment.prepare_experiment(self.source, self.output, label_additions=additions)
        self.assertEqual((self.output / "labels/train/macworx__box.txt").read_text(),
                         "4 .5 .5 .5 .5\n5 .2 .2 .2 .2\n")

    def test_bad_or_unmatched_additions_fail_before_creating_output(self):
        before = self.source_bytes()
        for image, rows in (("not_here.png", ["1 .5 .5 .4 .4"]),
                            ("food_0.png", ["99 .5 .5 .4 .4"]),
                            ("food_0.png", ["1 nan .5 .4 .4"]),
                            ("food_0.png", ["1 .95 .5 .4 .4"]),
                            ("food_0.png", ["1 .2 .2 .4 .4 .6 .6"]),
                            ("../food_0.png", ["1 .5 .5 .4 .4"])):
            with self.subTest(image=image, rows=rows):
                additions = self.additions([{"source": "meal", "image": image,
                                            "rows": rows, "provenance": "review"}])
                with self.assertRaises(ValueError):
                    experiment.prepare_experiment(self.source, self.output, label_additions=additions)
                self.assertFalse(self.output.exists())
                self.assertEqual(self.source_bytes(), before)

    def test_output_guards_preserve_sources_and_unrelated_files(self):
        unrelated = self.root / "notes"
        unrelated.mkdir()
        sentinel = unrelated / "keep.txt"
        sentinel.write_text("keep")
        for output in (self.source, self.source.parent, self.source / "nested",
                       self.root, self.root / "data/raw/new", self.root / "data/processed/new",
                       self.root / "model/new", unrelated):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "output"):
                experiment.prepare_experiment(self.source, output)
        self.assertEqual(sentinel.read_text(), "keep")
        self.assertTrue(self.source_yaml.exists())

    def test_invalid_balancing_options_do_not_create_output(self):
        for classes, maximum in ((["unknown"], 3), (["plate"], 0), (["plate"], 4)):
            with self.subTest(classes=classes, maximum=maximum), self.assertRaises(ValueError):
                experiment.prepare_experiment(self.source, self.output,
                                              balance_classes=classes, max_repeat=maximum)
            self.assertFalse(self.output.exists())

    def test_default_keeps_original_views_and_copies_labels_independently(self):
        experiment.prepare_experiment(self.source, self.output)
        self.assertEqual(self.summary()["counts"], {"train_originals": 8, "train_views": 8,
                                                    "val_originals": 2, "val_views": 2})
        generated = self.output / "labels/val/meal__occluded.txt"
        self.assertNotEqual(generated.stat().st_ino,
                            (self.source / "labels/val/meal__occluded.txt").stat().st_ino)
        generated.write_text("modified experiment only\n")
        self.assertEqual((self.source / "labels/val/meal__occluded.txt").read_text(),
                         "0 .5 .5 .4 .4\n1 .5 .5 .8 .8\n")

    def test_weak_subset_keeps_full_validation_and_all_labels(self):
        original = self.source_bytes()
        dataset = experiment.prepare_weak_subset(
            self.source, self.output, replay_fraction=0.20)
        config = yaml.safe_load(dataset.read_text())
        manifest = Path(config["train"])
        selected = [Path(s) for s in manifest.read_text().splitlines()]
        self.assertEqual(len(selected), 4)  # 3 weak originals + 1 of 5 strong-only.
        self.assertEqual(len(set(selected)), 4)
        for path in selected:
            self.assertTrue(path.is_file())
            self.assertTrue(experiment.label_path(path).is_file())
        self.assertEqual(Path(config["val"]).resolve(),
                         (self.source/"images/val").resolve())
        self.assertEqual(config["names"], train.class_names())
        self.assertEqual(self.source_bytes(), original)
        report = json.loads((self.output/"weak_subset.json").read_text())
        self.assertEqual(report["source_training_views"], 8)
        self.assertEqual(report["selected_training_views"], 4)
        self.assertEqual(report["weak_original_groups"], 3)
        self.assertEqual(report["replay_original_groups"], 1)
        self.assertTrue(report["all_class_labels_preserved"])

    def test_weak_subset_selects_all_original_and_crop_views_together(self):
        focus = self.root / "data/focus"
        experiment.prepare_experiment(
            self.source, focus, balance_classes=["plate", "wrapper", "bottle"],
            max_repeat=2, crop_repeats=True)
        dataset = experiment.prepare_weak_subset(
            focus, self.output, replay_fraction=0)
        files = [Path(v) for v in
                 Path(yaml.safe_load(dataset.read_text())["train"]).read_text().splitlines()]
        self.assertEqual(len(files), 6)  # Three weak originals, each with 2 views.
        self.assertTrue(all(p.parent == focus/"images/train" for p in files))
        for stem in ("meal__plate", "meal__wrapper", "meal__bottle"):
            self.assertEqual(sum(path.stem.startswith(stem) for path in files), 2)
        self.assertEqual(
            json.loads((self.output/"weak_subset.json").read_text())[
                "selected_training_views"], 6)

    def test_weak_subset_bad_parameters_do_not_write_output(self):
        for classes, fraction in ((["not_a_class"], .2), (["plate"], -0.1),
                                  ([], .2), (["wrapper"], 1.1)):
            with self.subTest(classes=classes, fraction=fraction), self.assertRaises(ValueError):
                experiment.prepare_weak_subset(
                    self.source, self.output, classes, fraction)
            self.assertFalse(self.output.exists())

    def test_relative_source_without_yaml_path_resolves_against_the_yaml_directory(self):
        self.source_yaml.write_text(yaml.safe_dump({"train": "images/train", "val": "images/val",
                                                    "names": train.class_names()}))
        original = Path.cwd()
        try:
            os.chdir(self.root)
            experiment.prepare_experiment(Path("data/training"), Path("data/training-revised"))
        finally:
            os.chdir(original)
        self.assertEqual(self.summary()["counts"]["val_views"], 2)


if __name__ == "__main__":
    unittest.main()
