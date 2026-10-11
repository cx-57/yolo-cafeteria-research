"""Fast synthetic tests for split integrity, 10x views, and annotation safety."""
from pathlib import Path
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train


class TrainingPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.processed = root / "processed"
        self.output = root / "training"
        self.patch_processed = patch.object(train, "DATA", root)
        self.patch_output = patch.object(train, "OUTPUT", self.output)
        self.patch_processed.start()
        self.patch_output.start()
        self.addCleanup(self.patch_processed.stop)
        self.addCleanup(self.patch_output.stop)

    def add_image(self, source, name, color, row="0 .5 .5 .4 .4"):
        image_dir = self.processed / source / "images"
        label_dir = self.processed / source / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        filename = image_dir / (name + ".png")
        Image.new("RGB", (36, 24), color).save(filename)
        (label_dir / (name + ".txt")).write_text(row + "\n")
        return filename

    def populate(self):
        for i in range(45):
            # Colors intentionally differ: don't group unrelated files by pixel.
            self.add_image("school_lunch", f"food_{i:02d}", (i * 4, 50, 75))

    def test_ten_views_per_training_original_and_untouched_validation(self):
        self.populate()
        first_label = (self.processed / "school_lunch/labels/food_00.txt").read_text()
        train.build_dataset(augment_factor=10, seed=42)
        result = json.loads((self.output / "split_summary.json").read_text())
        t, v = result["train_originals"], result["val_originals"]
        self.assertEqual(t + v, 45)
        self.assertEqual((t, v), (36, 9))  # Exact 80/20 originals, not 80/20 of eligible only.
        self.assertEqual(result["train_views"], 10 * t)
        self.assertEqual(result["test_originals"], 0)
        self.assertGreater(v, 0)
        self.assertEqual(len(list((self.output / "images/train").glob("*.png"))), t * 10)
        self.assertEqual(len(list((self.output / "images/val").glob("*.png"))), v)
        self.assertEqual(len(list((self.output / "labels/train").glob("*.txt"))), t * 10)
        self.assertFalse(any("__view" in p.stem for p in (self.output / "images/val").iterdir()))
        self.assertEqual((self.processed / "school_lunch/labels/food_00.txt").read_text(), first_label)
        original = next(p for p in (self.output / "images/train").iterdir() if "__view" not in p.stem)
        mirrored = original.with_name(original.stem + "__view01" + original.suffix)
        self.assertEqual(os.stat(original).st_ino, os.stat(mirrored).st_ino)

    def test_ratio_applies_to_all_sources_without_leaking_train_only_photos(self):
        self.populate()
        for i in range(15):
            self.add_image("food_american", f"additional_{i:02d}",
                           (i * 3, 100, 15))
        train.build_dataset(augment_factor=1, seed=42)
        result = json.loads((self.output / "split_summary.json").read_text())
        self.assertEqual(result["original_images"], 60)
        self.assertEqual(result["target_validation_originals"], 12)
        self.assertEqual(result["val_originals"], 12)
        self.assertEqual(result["train_originals"], 48)
        self.assertEqual(result["test_originals"], 0)
        self.assertFalse(any(p.name.startswith("food_american__")
                             for p in (self.output / "images/val").iterdir()))

    def test_deterministic_grouped_split(self):
        self.populate()
        train.build_dataset(augment_factor=1, seed=9)
        before = sorted(p.name for p in (self.output / "images/val").iterdir())
        train.build_dataset(augment_factor=1, seed=9)
        after = sorted(p.name for p in (self.output / "images/val").iterdir())
        self.assertEqual(before, after)
        self.assertEqual(train.group_for_split("school_lunch",
                         "train__photo.rf.abc"), train.group_for_split("school_lunch",
                         "valid__photo.rf.xyz"))

    def test_exact_duplicate_cross_source_cannot_enter_validation(self):
        self.populate()
        # New food source is forced train-only. Duplicate pixel with old source
        # must drag the old source original into training too.
        self.add_image("food_american", "duplicate", (0, 50, 75))
        train.build_dataset(augment_factor=1, seed=42)
        self.assertTrue((self.output / "images/train/school_lunch__food_00.png").is_file())
        self.assertTrue((self.output / "images/train/food_american__duplicate.png").is_file())

    def test_source_labels_are_converted_only_in_generated_copy(self):
        self.populate()
        image = self.add_image("macworx", "box", (200, 20, 80),
                               "5 .5 .5 .3 .3")  # Historical ID 5 -> current ID 4.
        train.build_dataset(augment_factor=1)
        label = list((self.output / "labels/train").glob("macworx__box.txt"))
        if not label:
            label = list((self.output / "labels/val").glob("macworx__box.txt"))
        self.assertTrue(label)
        self.assertTrue(label[0].read_text().startswith("4 "))
        self.assertTrue(image.with_name(image.name).exists())
        self.assertTrue((self.processed / "macworx/labels/box.txt").read_text().startswith("5 "))

    def test_invalid_geometry_fails_before_training(self):
        self.populate()
        self.add_image("food_american", "bad", (100, 200, 100), "0 .95 .5 .4 .4")
        with self.assertRaises(ValueError):
            train.build_dataset()

    def test_roundoff_correction_does_not_edit_source(self):
        self.populate()
        original = "0 0.500000 0.502695 1.000000 0.994629"
        self.add_image("food_american", "rounding", (109, 97, 32), original)
        train.build_dataset(augment_factor=1)
        generated = self.output / "labels/train/food_american__rounding.txt"
        self.assertTrue(generated.exists())  # New imports are train-only.
        parts = generated.read_text().split()
        x, y, w, h = map(float, parts[1:])
        self.assertLessEqual(y + h / 2, 1.000001)
        self.assertEqual((self.processed / "food_american/labels/rounding.txt").read_text(),
                         original + "\n")
        report = json.loads((self.output / "split_summary.json").read_text())
        self.assertEqual(report["minor_boundary_rounding_clips_in_generated_labels"]["food_american"], 1)

    def test_orphan_label_and_missing_image_fail(self):
        self.populate()
        self.add_image("food_american", "one", (100, 100, 100))
        (self.processed / "food_american/labels/orphan.txt").write_text("0 .5 .5 .2 .2\n")
        with self.assertRaisesRegex(ValueError, "Label without image"):
            train.build_dataset()

    def test_factor_one_and_invalid_factor(self):
        self.populate()
        with self.assertRaises(ValueError):
            train.build_dataset(augment_factor=0)
        train.build_dataset(augment_factor=1)
        report = json.loads((self.output / "split_summary.json").read_text())
        self.assertEqual(report["train_originals"], report["train_views"])


if __name__ == "__main__":
    unittest.main()
