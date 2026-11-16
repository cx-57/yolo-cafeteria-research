"""Add gated specialist detections while preserving every primary prediction."""
import math

from specialists import validate_prediction_rows
from v5_box_refinement import overlap


def recover_detections(base, expert, class_id, min_confidence=.65, max_area=.3,
                       base_confidence=.25, min_iou=.55, confidence_multiplier=.5):
    values = (min_confidence, max_area, base_confidence, min_iou, confidence_multiplier)
    if (type(class_id) is not int or not 0 <= class_id < 6
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
            or not 0 <= min_confidence <= 1 or not 0 < max_area <= 1
            or not 0 <= base_confidence <= 1 or not 0 < min_iou <= 1
            or not 0 < confidence_multiplier <= 1):
        raise ValueError('Invalid recovery settings')
    validate_prediction_rows(base)
    validate_prediction_rows(expert)
    result = [list(row) for row in base]
    blocking = [row for row in base if int(row[5]) == class_id and row[4] >= base_confidence]
    for row in sorted(expert, key=lambda row: -row[4]):
        if (int(row[5]) != class_id or row[4] < min_confidence
                or (row[2] - row[0]) * (row[3] - row[1]) > max_area):
            continue
        if any(overlap(row, previous) >= min_iou for previous in blocking):
            continue
        result.append([*row[:4], confidence_multiplier * row[4], class_id])
        blocking.append(row)
    return result
