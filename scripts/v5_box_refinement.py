"""Experimental box-only agreement refinement; no confidence or class selection."""
import math

from specialists import validate_prediction_rows


def overlap(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.


def refine_geometry(base, expert, class_id=5, min_iou=.55, min_confidence=.25, weight=.5):
    """Move agreed boxes, preserving detection count, order, confidence and class.

    Each expert box can refine at most one base box. Highest-confidence base
    boxes get first choice; unmatched expert detections are never appended.
    Inputs use normalized xyxy/confidence/global-class rows and stay unchanged.
    """
    if (type(class_id) is not int or not 0 <= class_id < 6
            or any(type(v) not in (int, float) or not math.isfinite(v)
                   for v in (min_iou, min_confidence, weight))
            or not 0 < min_iou <= 1 or not 0 <= min_confidence <= 1 or not 0 <= weight <= 1):
        raise ValueError('Invalid geometry-refinement settings')
    validate_prediction_rows(base)
    validate_prediction_rows(expert)
    result = [list(row) for row in base]
    available = {i for i, row in enumerate(expert)
                 if int(row[5]) == class_id and row[4] >= min_confidence}
    for index in sorted(range(len(base)), key=lambda i: (-base[i][4], i)):
        row = base[index]
        if int(row[5]) != class_id:
            continue
        score, match = max(((overlap(row, expert[j]), j) for j in sorted(available)),
                           default=(0., -1), key=lambda item: (item[0], -item[1]))
        if score >= min_iou:
            result[index][:4] = [(1 - weight) * row[k] + weight * expert[match][k] for k in range(4)]
            available.remove(match)
    return result
