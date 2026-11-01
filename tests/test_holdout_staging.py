"""Synthetic tests for provisional COCO review staging; no real image certification."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import stage_holdout_candidates as stage


class HoldoutStagingTests(unittest.TestCase):
    def test_exact_source_labels_are_only_review_hints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "source"
            images.mkdir()
            Image.new("RGB", (32, 32), "white").save(images / "a.png")
            Image.new("RGB", (32, 32), "black").save(images / "b.png")
            source = dict(categories=[
                dict(id=1, name="packet"), dict(id=2, name="Plastic Bottle"),
                dict(id=3, name="generic plastic container")],
                images=[dict(id=1, file_name="a.png"),
                        dict(id=2, file_name="b.png"),
                        dict(id=3, file_name="../unsafe.png")],
                annotations=[dict(image_id=1, category_id=1),
                             dict(image_id=2, category_id=2),
                             dict(image_id=3, category_id=3)])
            annotations = root / "annotations.json"
            annotations.write_text(json.dumps(source))
            found, categories = stage.collect(annotations, images)
            self.assertEqual(len(found), 2)
            self.assertEqual({tuple(row[1]) for row in found}, {("bottle",), ("wrapper",)})
            self.assertIn("generic plastic container", categories)
            self.assertNotIn("generic plastic container", stage.HINTS["hinged_container"])

    def test_cli_preserves_originals_and_never_marks_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            Image.new("RGB", (24, 24), "yellow").save(source / "first.png")
            data = dict(categories=[dict(id=1, name="spoon")],
                        images=[dict(id=1, file_name="first.png")],
                        annotations=[dict(image_id=1, category_id=1)])
            annotations = root / "annotations.json"
            annotations.write_text(json.dumps(data))
            before = (source / "first.png").read_bytes()
            out = root / "out"
            cmd = [sys.executable, str(Path(stage.__file__)), "--coco", str(annotations),
                   "--images-root", str(source), "--source", "Synthetic",
                   "--source-url", "https://example.org", "--license", "example",
                   "--out", str(out)]
            run = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertTrue((out / "contact-sheet.jpg").is_file())
            self.assertEqual(before, (source / "first.png").read_bytes())
            with (out / "review.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source_hints"], "utensil")
            self.assertEqual(rows[0]["visual_status"], "PENDING")
            self.assertEqual(rows[0]["independent"], "NO")
            self.assertEqual(rows[0]["ground_truth"], "NOT_LABELED")
            self.assertEqual(subprocess.run(cmd, capture_output=True).returncode != 0, True)


if __name__ == "__main__":
    unittest.main()
