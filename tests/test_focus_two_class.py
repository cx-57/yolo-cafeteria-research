"""Unit tests for the non-destructive two-class dataset conversion."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from focus_two_class import build, evaluate, two_class_rows, verify


class TwoClassTests(unittest.TestCase):
    def test_build_from_relocated_reviewed_source_preserves_historical_config(self):
        from PIL import Image
        import yaml
        from focus_three_class import audit, fingerprint, pixels, verify as verify_three
        from v2 import digest
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp).resolve() / "relocated-review"
            rows = ["0 0.2 0.2 0.1 0.1", "1 0.5 0.5 0.1 0.1", "2 0.8 0.8 0.1 0.1"]
            records = []
            for split, color, labels in [("train", "red", rows),
                                         ("train", "green", [rows[1]]), ("val", "blue", rows)]:
                name = f"fixture__{color}.jpg"
                image = source / "images" / split / name
                label = source / "labels" / split / (Path(name).stem + ".txt")
                image.parent.mkdir(parents=True, exist_ok=True)
                label.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (20, 20), color).save(image)
                label.write_text("\n".join(labels))
                records.append(dict(name=name, split=split, original_split=split,
                                    group=f"fixture__{color}.jpg", sha=digest(image),
                                    pixels=pixels(image), rows=labels, verified_absence=False,
                                    review_status="signed_off", input_label_sha256=digest(label),
                                    override_applied=False))
            config = dict(path="/previous-checkout/reviewed", train="images/train",
                          val="images/val", names=["plate", "wrapper", "bottle"])
            (source / "data.yaml").write_text(yaml.safe_dump(config))
            original_config = (source / "data.yaml").read_bytes()
            (source / "manifest.json").write_text(json.dumps(dict(
                selected=3, records=records, counts=audit(records, 3), fingerprint=fingerprint(records))))
            output = source.parent / "two-class"
            try:
                build(source, output)
            except ValueError as error:
                self.fail(f"Read-only derivation must accept a relocated reviewed source: {error}")
            self.assertEqual(verify(output)["counts"]["train"]["negatives"], 1)
            self.assertEqual((source / "data.yaml").read_bytes(), original_config)
            with self.assertRaisesRegex(ValueError, "Dataset YAML changed"):
                verify_three(source)
            config["train"] = "unrelated/images"
            (source / "data.yaml").write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, "Dataset YAML changed"):
                build(source, source.parent / "invalid-layout")

    def evaluate_metrics(self, box):
        """Exercise report validation without running the external detector."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            weights = root / "model.pt"
            weights.write_bytes(b"saved-two-class-checkpoint")
            output = root / "results"
            manifest = dict(source_fingerprint="reviewed-900", counts={"val": dict(
                images=126, objects={"plate": 137, "bottle": 45},
                class_images={"plate": 86, "bottle": 36})})
            metrics = SimpleNamespace(box=box, nt_per_class=[137, 45])
            def validate(**kwargs):
                (output / "evaluation").mkdir(parents=True)
                return metrics
            model = SimpleNamespace(names={0: "plate", 1: "bottle"}, val=validate)
            with patch("focus_two_class.verify", return_value=manifest), patch.dict(
                    sys.modules, {"ultralytics": SimpleNamespace(YOLO=lambda _: model)}):
                evaluate(root / "data", output, weights, "cpu", "evaluation")
            return json.loads((output / "evaluation/metrics.json").read_text())

    @staticmethod
    def valid_box():
        return SimpleNamespace(ap_class_index=[0, 1], ap50=[.92, .86],
                               ap=[.65, .55], p=[.9, .8], r=[.8, .7],
                               map50=.89, map=.6, mp=.85, mr=.75)

    def test_two_class_evaluation_rejects_nonfinite_metrics(self):
        box = self.valid_box()
        box.ap50[1] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite|invalid"):
            self.evaluate_metrics(box)

    def test_two_class_evaluation_rejects_extra_ap_rows(self):
        box = self.valid_box()
        box.ap50.append(.99)
        with self.assertRaisesRegex(ValueError, "rows|align"):
            self.evaluate_metrics(box)

    def test_two_class_evaluation_rejects_inconsistent_macro(self):
        box = self.valid_box()
        box.map50 = .99
        with self.assertRaisesRegex(ValueError, "macro"):
            self.evaluate_metrics(box)

    def test_two_class_report_has_support_f1_and_checkpoint_identity(self):
        report = self.evaluate_metrics(self.valid_box())
        self.assertIn("objects", report["per_class"]["bottle"])
        self.assertEqual(report["per_class"]["bottle"]["objects"], 45)
        self.assertAlmostEqual(report["per_class"]["bottle"]["F1"], 2 * .8 * .7 / 1.5)
        self.assertEqual(report["weights_sha256"], hashlib.sha256(b"saved-two-class-checkpoint").hexdigest())
        self.assertFalse(report["independent_test"])

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
