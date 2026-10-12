"""Gradient and real upstream-loss checks for partially annotated sources."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

try:
    import torch
    from torch import nn
except ImportError:
    # The fast GitHub Actions job intentionally installs only Pillow and PyYAML.
    # Integration tests are exercised on machines with the pinned ML stack.
    torch = None
    nn = SimpleNamespace(Module=object)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    partial = importlib.import_module("partial_labels")
except ModuleNotFoundError as error:
    if error.name not in ("torch", "ultralytics", "partial_labels"):
        raise
    partial = None

NAMES = ["food", "plate", "paper_boat", "utensil", "hinged_container", "napkin",
         "drink_carton", "condiment_packet", "wrapper", "bottle", "bowl"]


@unittest.skipUnless(torch is not None and partial is not None,
                     "Requires torch and pinned ultralytics (run locally for integration)")
class PartialLabelsTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(partial, "The partial-annotation loss implementation is missing")
        self.policy_data = {"schema_version": 1, "names": NAMES,
                            "source_known_classes": {"plate": [1], "school_lunch": [0, 6]}}
        self.policy = partial.CoveragePolicy.from_dict(self.policy_data, NAMES)

    def masked_bce(self, logits, targets, known):
        loss = partial.CoverageMaskedBCE()
        loss.set_known_mask(known)
        return loss(logits, targets)

    def test_all_known_classes_equal_standard_bce_and_gradients(self):
        logits = torch.linspace(-2, 2, 44).reshape(2, 2, 11).requires_grad_()
        targets = torch.zeros_like(logits)
        targets[0, 0, 0] = .3
        targets[1, 1, 10] = .9
        actual = self.masked_bce(logits, targets, torch.ones(2, 1, 11, dtype=torch.bool))
        expected = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        a = torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0]
        b = torch.autograd.grad(expected.sum(), logits)[0]
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_unknown_negative_has_zero_gradient_but_unknown_soft_positive_is_kept(self):
        logits = torch.zeros(1, 2, 11, requires_grad=True)
        targets = torch.zeros_like(logits)
        targets[0, 1, 0] = .75
        known = self.policy.known_mask(["plate__photo.jpg"], 1, "cpu")
        losses = self.masked_bce(logits, targets, known)
        losses.sum().backward()
        self.assertEqual(float(losses[0, 0, 0]), 0)
        self.assertEqual(float(logits.grad[0, 0, 0]), 0)
        self.assertGreater(float(losses[0, 1, 0]), 0)
        self.assertAlmostEqual(float(logits.grad[0, 1, 0]), -.25)

    def test_known_plate_hard_negative_retains_penalty(self):
        logits = torch.zeros(1, 1, 11, requires_grad=True)
        targets = torch.zeros_like(logits)
        known = self.policy.known_mask(["plate__chair.jpg"], 1, "cpu")
        losses = self.masked_bce(logits, targets, known)
        losses.sum().backward()
        self.assertGreater(float(losses[0, 0, 1]), 0)
        self.assertEqual(float(logits.grad[0, 0, 1]), .5)
        self.assertEqual(float(logits.grad[0, 0, 0]), 0)

    def test_batch_order_and_balanced_view_names_keep_source_coverage(self):
        known = self.policy.known_mask([
            "/tmp/train/school_lunch__IMG_1__view03.jpg",
            "/tmp/train/plate__train__lvis_42__weakview02.jpg"], 2, "cpu")
        self.assertEqual(tuple(known.shape), (2, 1, 11))
        self.assertEqual(known[0, 0].nonzero().flatten().tolist(), [0, 6])
        self.assertEqual(known[1, 0].nonzero().flatten().tolist(), [1])

    def test_missing_unknown_or_misaligned_image_provenance_is_rejected(self):
        for files, batch in [(None, 1), (["photo.jpg"], 1),
                             (["unregistered__photo.jpg"], 1), (["plate__photo.jpg"], 2)]:
            with self.subTest(files=files), self.assertRaises(ValueError):
                self.policy.known_mask(files, batch, "cpu")

    def test_mismatched_ontology_and_invalid_coverage_ids_are_rejected(self):
        bad = dict(self.policy_data, names=list(reversed(NAMES)))
        with self.assertRaises(ValueError):
            partial.CoveragePolicy.from_dict(bad, NAMES)
        for ids in [[11], [-1], [1, 1], [True], [1.0], "plate"]:
            bad = dict(self.policy_data, source_known_classes={"plate": ids})
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                partial.CoveragePolicy.from_dict(bad, NAMES)
        with self.assertRaises(ValueError):
            self.policy.validate_names({i: name for i, name in enumerate(reversed(NAMES))})

    def test_policy_file_requires_explicit_schema_and_source_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coverage.json"
            path.write_text(json.dumps(self.policy_data))
            loaded = partial.CoveragePolicy.from_file(path, NAMES)
            self.assertEqual(loaded.known_mask(["plate__x.jpg"], 1, "cpu").sum().item(), 1)
            for field in ["schema_version", "names", "source_known_classes"]:
                bad = dict(self.policy_data)
                del bad[field]
                path.write_text(json.dumps(bad))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    partial.CoveragePolicy.from_file(path, NAMES)

    def test_mixed_image_augmentation_is_rejected(self):
        valid = dict(mosaic=0, mixup=0, cutmix=0, copy_paste=0)
        partial.validate_training_options(SimpleNamespace(**valid))
        for option in valid:
            bad = dict(valid, **{option: .1})
            with self.subTest(option=option), self.assertRaises(ValueError):
                partial.validate_training_options(SimpleNamespace(**bad))

    def test_bce_cannot_silently_use_missing_or_wrong_batch_mask(self):
        loss = partial.CoverageMaskedBCE()
        logits, targets = torch.zeros(2, 3, 11), torch.zeros(2, 3, 11)
        with self.assertRaises(ValueError):
            loss(logits, targets)
        loss.set_known_mask(torch.ones(1, 1, 11, dtype=torch.bool))
        with self.assertRaises(ValueError):
            loss(logits, targets)


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.nc, self.reg_max = 11, 4
        self.stride = torch.tensor([8.])


class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.model = nn.ModuleList([TinyHead()])
        self.args = SimpleNamespace(box=7.5, cls=.5, dfl=1.5)
        self.names = dict(enumerate(NAMES))


@unittest.skipUnless(torch is not None and partial is not None,
                     "Requires torch and pinned ultralytics")
class UpstreamLossTests(PartialLabelsTests):
    def setUp(self):
        super().setUp()
        import ultralytics
        if ultralytics.__version__ != "8.4.175":
            self.skipTest("Integration targets the inspected Ultralytics 8.4.175 API")
        from ultralytics.utils.loss import v8DetectionLoss
        self.standard = v8DetectionLoss
        torch.manual_seed(10)
        self.model = TinyDetector()
        # The upstream task-aligned assigner selects top-10 candidates, so
        # a 2x2 (= 4 anchor) synthetic map is invalid. Use 4x4 (= 16 anchors).
        self.preds = {"boxes": torch.randn(2, 16, 4, requires_grad=True),
                      "scores": torch.randn(2, 11, 16, requires_grad=True),
                      "feats": [torch.zeros(2, 1, 4, 4)]}
        self.batch = {"batch_idx": torch.tensor([0., 1.]), "cls": torch.tensor([[0.], [1.]]),
                      "bboxes": torch.tensor([[.5, .5, .75, .75], [.5, .5, .5, .5]]),
                      "im_file": ["plate__food.jpg", "school_lunch__plate.jpg"]}

    def test_actual_yolo11n_forward_backward_on_a_real_plate_photo(self):
        """Real architecture + committed photo, but UNTRAINED weights: runtime smoke only."""
        import numpy as np
        from PIL import Image
        from ultralytics.nn.tasks import DetectionModel
        torch.manual_seed(42)
        torch.set_num_threads(2)
        model = DetectionModel(cfg="yolo11n.yaml", nc=11, verbose=False)
        model.names = dict(enumerate(NAMES))
        model.train()
        photo = ROOT / "data/processed/plate/images/train__lvis_000000225490.jpg"
        with Image.open(photo) as original:
            rgb = original.convert("RGB").resize((128, 128))
            pixels = np.asarray(rgb).copy()
        image = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).float() / 255
        batch = {"batch_idx": torch.tensor([0.]), "cls": torch.tensor([[1.]]),
                 "bboxes": torch.tensor([[.5, .5, .5, .5]]),
                 "im_file": ["plate__train__lvis_000000225490.jpg"]}
        predictions = model(image)
        criterion = partial.make_criterion(model, self.policy)
        loss, components = criterion(predictions, batch)
        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue(torch.isfinite(torch.stack(list(components.values()))).all())
        loss.sum().backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        self.assertIsNone(criterion.bce.known_mask)

    def test_all_known_full_upstream_detection_loss_is_unchanged(self):
        data = dict(self.policy_data, source_known_classes={
            "plate": list(range(11)), "school_lunch": list(range(11))})
        policy = partial.CoveragePolicy.from_dict(data, NAMES)
        actual = partial.make_criterion(self.model, policy)(self.preds, self.batch)
        expected = self.standard(self.model)(self.preds, self.batch)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        for name in ["box_loss", "cls_loss", "dfl_loss"]:
            torch.testing.assert_close(actual[1][name], expected[1][name], rtol=0, atol=0)

    def test_partial_loss_preserves_box_and_dfl_components_and_box_gradients(self):
        criterion = partial.make_criterion(self.model, self.policy)
        actual = criterion(self.preds, self.batch)
        expected = self.standard(self.model)(self.preds, self.batch)
        for name in ["box_loss", "dfl_loss"]:
            torch.testing.assert_close(actual[1][name], expected[1][name], rtol=0, atol=0)
        a = torch.autograd.grad(actual[0].sum(), self.preds["boxes"], retain_graph=True)[0]
        b = torch.autograd.grad(expected[0].sum(), self.preds["boxes"])[0]
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertLess(actual[1]["cls_loss"], expected[1]["cls_loss"])
        self.assertIsNone(criterion.bce.known_mask)


if __name__ == "__main__":
    unittest.main()
