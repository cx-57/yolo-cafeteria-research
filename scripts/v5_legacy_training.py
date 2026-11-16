"""Preserve the frozen V1 utensil architecture for separate fine-tuning trials."""
from copy import deepcopy

import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer


def preserve_utensil_model(source):
    """Copy actual modules; rebuilding from YAML silently loses the custom head."""
    names = getattr(source, 'names', {})
    ordered = list(names.values()) if isinstance(names, dict) else list(names)
    if ordered != ['utensil'] or not hasattr(source, 'model'):
        raise ValueError('Expected the native one-class V1 utensil detector')
    head = source.model[-1]
    if (getattr(head, 'nc', None) != 1 or not hasattr(head, 'cv2') or not hasattr(head, 'cv3')
            or not any(isinstance(m, torch.nn.MaxPool3d) for m in head.cv3.modules())):
        raise ValueError('Expected the preserved V1 three-logit pooling head')
    # This keeps all original builtin module types, so saved models need no new
    # custom model class. Contiguous NCHW is required by the legacy pooling.
    return deepcopy(source).float().to(memory_format=torch.contiguous_format)


class PreservedUtensilTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        names = self.data['names']
        ordered = list(names.values()) if isinstance(names, dict) else list(names)
        if self.data['nc'] != 1 or ordered != ['utensil'] or weights is None:
            raise ValueError('Legacy fine-tuning requires native utensil data and loaded V1 modules')
        return preserve_utensil_model(weights)
