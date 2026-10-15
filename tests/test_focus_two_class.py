"""Unit tests for the non-destructive two-class dataset conversion."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from focus_two_class import build, two_class_rows, verify


class TwoClassTests(unittest.TestCase):
    def test_wrapper_removed_and_bottle_remapped_without_touching_coordinates(self):
        rows = [
            "0 0.5 0.5 0.1 0.2",
            "1 0.1 0.2 0.3 0.4",
            "2 0.3 0.4 0.2 0.1",
        ]
        self.assertEqual(two_class_rows(rows), [
            "0 0.5 0.5 0.1 0.2",
            "1 0.3 0.4 0.2 0.1",
        ])
        self.assertEqual(two_class_rows([rows[1]]), [])
        with self.assertRaises(ValueError):
            two_class_rows(["9 0.3 0.4 0.2 0.1"])

    def test_split_and_reviewed_negatives_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "reviewed-three-class"
            output = root / "derived-two-class"
            records = []
            cases = [
                ("train", "a.jpg", ["0 0.5 0.5 0.2 0.2"]),
                ("train", "b.jpg", ["1 0.5 0.5 0.2 0.2"]),
                ("train", "c.jpg", ["2 0.5 0.5 0.2 0.2"]),
                ("val", "d.jpg", ["0 0.5 0.5 0.2 0.2"]),
                ("val", "e.jpg", ["2 0.5 0.5 0.2 0.2"]),
            ]
            for split, name, rows in cases:
                folder = source / "images" / split
                folder.mkdir(parents=True, exist_ok=True)
                data = (split + ":" + name).encode()
                (folder / name).write_bytes(data)
                records.append(dict(split=split, name=name, rows=rows,
                                    sha=hashlib.sha256(data).hexdigest()))
            signed_off = dict(records=records, fingerprint="approved-three-class-hash")
            with patch("focus_two_class.verify_three", return_value=signed_off):
                build(source, output)
                manifest = verify(output)
                self.assertEqual(manifest["counts"]["train"]["objects"]["plate"], 1)
                self.assertEqual(manifest["counts"]["train"]["objects"]["bottle"], 1)
                self.assertEqual(manifest["counts"]["val"]["objects"]["bottle"], 1)
                self.assertEqual(manifest["counts"]["train"]["negatives"], 1)
                self.assertEqual((output / "labels/train/b.txt").read_text(), "")
                self.assertEqual((output / "labels/train/c.txt").read_text(),
                                 "1 0.5 0.5 0.2 0.2\n")
                self.assertEqual((source / "images/train/b.jpg").read_bytes(), b"train:b.jpg")
                with self.assertRaises(ValueError):
                    build(source, output)
                (output / "labels/train/c.txt").write_text("2 0.5 0.5 0.2 0.2\n")
                with self.assertRaisesRegex(ValueError, "Altered image or labels"):
                    verify(output)

    def test_incomplete_unapproved_source_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch("focus_two_class.verify_three", side_effect=ValueError("Source failed audit")):
                with self.assertRaisesRegex(ValueError, "Source failed audit"):
                    build(root / "source", root / "new-data")
                self.assertFalse((root / "new-data").exists())


if __name__ == "__main__":
    unittest.main()
