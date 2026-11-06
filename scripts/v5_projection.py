"""Remove the final bottle output while preserving a trained YOLO11 detector."""
import copy
import math

from v5_dataset import NAMES


def without_bottle(source):
    names = source.names
    ordered = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
    if ordered != NAMES + ['bottle']:
        raise ValueError('Expected the exact original V5 seven-class order')
    return project_classes(source, NAMES)


def project_classes(source, retained):
    import torch.nn as nn
    names = source.names
    ordered = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
    if (ordered not in (NAMES, NAMES + ['bottle']) or not retained
            or len(set(retained)) != len(retained) or not set(retained) <= set(NAMES)):
        raise ValueError('Invalid retained classes or unsupported source class order')
    indices = [ordered.index(name) for name in retained]
    result = copy.deepcopy(source)
    head = result.model[-1]
    if head.nc != len(ordered) or not hasattr(head, 'cv3'):
        raise ValueError('Unsupported original detection head')
    for name in ('cv3', 'one2one_cv3'):
        branches = getattr(head, name, None)
        if branches is None:
            continue
        for branch in branches:
            conv = branch[-1]
            if not isinstance(conv, nn.Conv2d) or conv.out_channels != len(ordered):
                raise ValueError('Unexpected classification output; refusing partial projection')
            replacement = nn.Conv2d(conv.in_channels, len(indices), conv.kernel_size,
                                    stride=conv.stride, padding=conv.padding,
                                    dilation=conv.dilation, groups=conv.groups,
                                    bias=conv.bias is not None, padding_mode=conv.padding_mode)
            replacement = replacement.to(device=conv.weight.device, dtype=conv.weight.dtype)
            replacement.weight = nn.Parameter(conv.weight[indices].detach().clone(),
                                              requires_grad=conv.weight.requires_grad)
            if conv.bias is not None:
                replacement.bias = nn.Parameter(conv.bias[indices].detach().clone(),
                                                requires_grad=conv.bias.requires_grad)
            branch[-1] = replacement
    head.nc = len(indices)
    head.no = head.reg_max * 4 + len(indices)
    result.names = dict(enumerate(retained))
    result.yaml = copy.deepcopy(result.yaml)
    result.yaml['nc'] = len(indices)
    return result


def assess_refinement(baseline, candidate):
    for result in (baseline, candidate):
        if set(result) != set(NAMES):
            raise ValueError('Comparison requires all six retained classes')
        for values in result.values():
            for key in ('AP50', 'AP50_95'):
                value = values.get(key)
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError('Comparison metrics must be finite probabilities')
    weak = [n for n in NAMES if baseline[n]['AP50'] < .8 or baseline[n]['AP50_95'] < .5] or NAMES
    deltas = {n: {k: candidate[n][k] - baseline[n][k] for k in ('AP50', 'AP50_95')} for n in NAMES}
    regressions = [n for n in NAMES if any(v < -.02 - 1e-12 for v in deltas[n].values())]
    weak_gain = sum(deltas[n]['AP50_95'] for n in weak) / len(weak)
    overall_gain = sum(deltas[n]['AP50_95'] for n in NAMES) / 6
    return dict(accept=not regressions and weak_gain >= .01 and overall_gain >= 0,
                initially_weak_classes=weak, weak_class_mean_AP50_95_gain=weak_gain,
                overall_AP50_95_gain=overall_gain, classes_regressing_more_than_two_points=regressions,
                per_class_deltas=deltas, scope='development validation only; prior checkpoint exposure unresolved')
