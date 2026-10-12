"""Opt-in negative-class masking for the inspected Ultralytics 8.4.175 API.

Source coverage is provisional, not certification that every object is boxed.
Only unknown *negative* BCE terms are removed. Assigned soft positives, target
assignment, box/DFL losses, gains, and their normalization stay upstream.

Usage::

    trainer = make_trainer("coverage.json")
    YOLO(weights).train(trainer=trainer, data=data_yaml, mosaic=0, mixup=0,
                        cutmix=0, copy_paste=0, workers=0, device="mps")

Policy JSON contains schema_version=1, names in current class-ID order, and
source_known_classes mapping source names to class-ID lists. Generated image
names must retain their source__ prefix, including any repeated training views.
The live training model receives the custom criterion; EMA, inference, saved
models, and validation/AP use standard Ultralytics models. A policy snapshot is
saved beside the run. Resume with the same factory/policy explicitly: ordinary
Ultralytics resume does not reconstruct this experimental training loss.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from collections.abc import Mapping

import torch
from torch import nn
import yaml
from ultralytics.utils.loss import v8DetectionLoss

SUPPORTED_ULTRALYTICS = "8.4.175"
MIXED_IMAGE_AUGMENTATIONS = ("mosaic", "mixup", "cutmix", "copy_paste")


def ontology_names():
    """Read the repository ontology without importing or changing train.py."""
    path = Path(__file__).resolve().parents[1] / "config.yaml"
    classes = yaml.safe_load(path.read_text())["classes"]
    if not isinstance(classes, dict) or set(classes) != set(range(len(classes))):
        raise ValueError("config.yaml requires contiguous integer class IDs")
    return [classes[i] for i in range(len(classes))]


@dataclass(frozen=True)
class CoveragePolicy:
    names: tuple
    source_known_classes: dict

    @classmethod
    def from_dict(cls, data, expected_names=None):
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise ValueError("Coverage policy requires schema_version=1")
        names = data.get("names")
        expected = list(expected_names if expected_names is not None else ontology_names())
        if (not isinstance(names, list) or names != expected
                or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("Coverage policy names do not match the current ontology")
        sources = data.get("source_known_classes")
        if not isinstance(sources, dict) or not sources:
            raise ValueError("Coverage policy requires source_known_classes metadata")
        checked = {}
        for source, ids in sources.items():
            if (not isinstance(source, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", source)
                    or not isinstance(ids, list)
                    or any(type(i) is not int or not 0 <= i < len(names) for i in ids)
                    or len(set(ids)) != len(ids)):
                raise ValueError(f"Invalid known-class coverage for source {source!r}")
            checked[source] = tuple(ids)
        return cls(tuple(names), checked)

    @classmethod
    def from_file(cls, path, expected_names=None):
        return cls.from_dict(json.loads(Path(path).read_text()), expected_names)

    def to_dict(self):
        return {"schema_version": 1, "names": list(self.names),
                "source_known_classes": {source: list(ids)
                                         for source, ids in self.source_known_classes.items()}}

    def validate_names(self, names):
        if isinstance(names, Mapping):
            if set(names) != set(range(len(self.names))):
                raise ValueError("Model/data class IDs do not match coverage policy")
            names = [names[i] for i in range(len(self.names))]
        if not isinstance(names, (list, tuple)) or tuple(names) != self.names:
            raise ValueError("Model/data names do not match coverage policy ontology")

    def known_mask(self, im_files, batch_size, device):
        if (not isinstance(im_files, (list, tuple)) or len(im_files) != batch_size
                or batch_size < 1):
            raise ValueError("Every batch image requires aligned im_file provenance")
        mask = torch.zeros(batch_size, 1, len(self.names), dtype=torch.bool, device=device)
        for row, filename in enumerate(im_files):
            if not isinstance(filename, (str, Path)):
                raise ValueError("Image provenance must be a generated source__ filename")
            name = Path(filename).name
            if "__" not in name:
                raise ValueError(f"Missing source__ prefix in image provenance: {name}")
            source, remainder = name.split("__", 1)
            if not remainder or source not in self.source_known_classes:
                raise ValueError(f"Unknown source in image provenance: {name}")
            mask[row, 0, list(self.source_known_classes[source])] = True
        return mask


class CoverageMaskedBCE(nn.Module):
    """Unreduced upstream BCE with per-image class coverage and soft positives."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.known_mask = None

    def set_known_mask(self, known_mask):
        if known_mask is not None and (not isinstance(known_mask, torch.Tensor)
                                       or known_mask.dtype != torch.bool):
            raise ValueError("Known-class mask must be a boolean tensor")
        self.known_mask = known_mask

    def forward(self, logits, target_scores):
        if (logits.ndim != 3 or logits.shape != target_scores.shape
                or self.known_mask is None
                or self.known_mask.shape != (logits.shape[0], 1, logits.shape[2])
                or self.known_mask.device != logits.device):
            raise ValueError("BCE requires an aligned batch-by-one-by-class coverage mask")
        keep = self.known_mask | (target_scores > 0)
        return self.bce(logits, target_scores) * keep


def require_compatible_api():
    from ultralytics import __version__
    if __version__ != SUPPORTED_ULTRALYTICS:
        raise ValueError(f"Coverage-aware loss requires inspected Ultralytics {SUPPORTED_ULTRALYTICS}; "
                         f"installed version is {__version__}")


class CoverageAwareDetectionLoss(v8DetectionLoss):
    def __init__(self, model, policy):
        require_compatible_api()
        policy.validate_names(model.names)
        head = model.model[-1]
        if head.nc != len(policy.names) or getattr(head, "one2one_cv2", None) is not None:
            raise ValueError("Coverage-aware loss requires the ordinary 11-class detection head")
        super().__init__(model)
        self.policy = policy
        self.bce = CoverageMaskedBCE()

    def __call__(self, preds, batch):
        parsed = self.parse_output(preds)
        scores = parsed["scores"]
        known = self.policy.known_mask(batch.get("im_file"), scores.shape[0], scores.device)
        self.bce.set_known_mask(known)
        try:
            return super().__call__(preds, batch)
        finally:
            self.bce.set_known_mask(None)


def make_criterion(model, policy):
    return CoverageAwareDetectionLoss(model, policy)


def validate_training_options(options):
    def value(name, default=None):
        return options.get(name, default) if isinstance(options, Mapping) else getattr(options, name, default)

    for name in MIXED_IMAGE_AUGMENTATIONS:
        probability = value(name, 0)
        if (not isinstance(probability, (int, float)) or not math.isfinite(probability)
                or probability != 0):
            raise ValueError(f"Coverage-aware loss requires {name}=0: mixed images lose source provenance")
    if value("single_cls", False) or value("distill_model") is not None or value("compile", False):
        raise ValueError("Coverage-aware experiment does not support single_cls, distillation, or compile")


def make_trainer(policy_path):
    """Return a single-device trainer class bound to an explicit policy file."""
    require_compatible_api()
    policy = CoveragePolicy.from_file(policy_path)
    from ultralytics.cfg import get_cfg
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils import DEFAULT_CFG
    from ultralytics.utils.torch_utils import unwrap_model

    class CoverageAwareTrainer(DetectionTrainer):
        def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
            validate_training_options(get_cfg(cfg, overrides))
            super().__init__(cfg, overrides, _callbacks)
            if self.world_size > 1:
                raise ValueError("Coverage-aware trainer factory supports one device, not spawned DDP")
            policy.validate_names(self.data["names"])
            self.coverage_policy = policy
            (self.save_dir / "coverage_policy.json").write_text(
                json.dumps(policy.to_dict(), indent=2) + "\n")

        def _setup_train(self):
            super()._setup_train()
            # Install after device placement and class weights. Upstream EMA was
            # already made from an ordinary model, so checkpoints need no custom
            # class and validation loss retains the standard comparison recipe.
            validate_training_options(self.args)
            model = unwrap_model(self.model)
            model.criterion = make_criterion(model, self.coverage_policy)
            if self.ema is not None:
                self.ema.ema.criterion = None

    return CoverageAwareTrainer
