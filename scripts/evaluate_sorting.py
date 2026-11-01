"""Evaluate disposal-bin decisions against independently labeled object boxes.

Requires the JSON produced by 'run_specialists.sh sort' and a human-labeled
ground-truth JSON file. This is not a substitute for independent labels.
"""
import argparse
import json
import math
from pathlib import Path

BINS = ("compost", "recycling", "landfill")
OUTPUTS = (*BINS, "review")


def valid_box(box):
    return (isinstance(box, list) and len(box) == 4
            and all(type(x) in (float, int) and math.isfinite(x) for x in box)
            and 0 <= box[0] < box[2] <= 1
            and 0 <= box[1] < box[3] <= 1)


def iou(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    overlap = width * height
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return overlap / (area_a + area_b - overlap)


def fraction(num, den):
    return num / den if den else None


def evaluate(prediction_data, truth_data, minimum_iou=0.5):
    """Match by geometry and confidence, never by the desired bin label."""
    if type(minimum_iou) not in (float, int) or not 0 < minimum_iou <= 1:
        raise ValueError("minimum_iou must be in (0, 1]")
    decisions = prediction_data.get("decisions")
    images = truth_data.get("images")
    if not isinstance(decisions, dict) or not isinstance(images, dict) or not images:
        raise ValueError("Expected nonempty truth.images and prediction.decisions dictionaries")
    if set(decisions) != set(images):
        raise ValueError("Image keys must match exactly; do not omit test photos or negatives")

    confusion = {actual: {predicted: 0 for predicted in (*OUTPUTS, "missed")}
                 for actual in BINS}
    extras = {predicted: 0 for predicted in OUTPUTS}
    total = correct = wrong = review = missed = 0

    for image_name in sorted(images):
        objects = images[image_name]
        predictions = decisions[image_name]
        if not isinstance(objects, list) or not isinstance(predictions, list):
            raise ValueError("Each image requires object and prediction lists")
        for obj in objects:
            if (not isinstance(obj, dict) or obj.get("category") not in BINS
                    or not valid_box(obj.get("box_xyxy_normalized"))):
                raise ValueError("Each truth object needs a valid box and independently checked bin")
        for pred in predictions:
            if (not isinstance(pred, dict) or pred.get("category") not in OUTPUTS
                    or not valid_box(pred.get("box_xyxy_normalized"))
                    or type(pred.get("detection_confidence")) not in (int, float)
                    or not math.isfinite(pred["detection_confidence"])
                    or not 0 <= pred["detection_confidence"] <= 1):
                raise ValueError("Invalid disposal prediction or confidence")

        total += len(objects)
        unmatched = set(range(len(objects)))
        # High confidence first; one prediction can match at most one object.
        ordered = sorted(predictions, key=lambda p: p["detection_confidence"], reverse=True)
        for pred in ordered:
            overlaps = [(iou(pred["box_xyxy_normalized"], objects[j]["box_xyxy_normalized"]), j)
                        for j in sorted(unmatched)]
            if not overlaps:
                extras[pred["category"]] += 1
                continue
            score, match = max(overlaps, key=lambda pair: (pair[0], -pair[1]))
            if score < minimum_iou:
                extras[pred["category"]] += 1
                continue
            unmatched.remove(match)
            actual = objects[match]["category"]
            predicted = pred["category"]
            confusion[actual][predicted] += 1
            if predicted == "review":
                review += 1
            elif predicted == actual:
                correct += 1
            else:
                wrong += 1
        for j in unmatched:
            confusion[objects[j]["category"]]["missed"] += 1
        missed += len(unmatched)

    false_auto = sum(extras[b] for b in BINS)
    automatic = correct + wrong
    return {
        "images": len(images),
        "ground_truth_objects": total,
        "minimum_box_iou": minimum_iou,
        "correct_automatic": correct,
        "wrong_bin_automatic": wrong,
        "matched_review": review,
        "missed_objects": missed,
        "unmatched_predictions": extras,
        "automatic_coverage_of_objects": fraction(automatic, total),
        "accuracy_among_matched_automatic_decisions": fraction(correct, automatic),
        "correct_automatic_over_all_labeled_objects": fraction(correct, total),
        "precision_including_unmatched_auto_predictions": fraction(
            correct, automatic + false_auto),
        "review_rate_of_labeled_objects": fraction(review, total),
        "miss_rate_of_labeled_objects": fraction(missed, total),
        "confusion_by_true_bin": confusion,
        "warning": ("These scores are only as reliable as independently assigned bin labels. "
                    "Previously trained-on or repeatedly inspected photos are not an independent test. "
                    "Review is abstention, not a correct prediction. "
                    "Unmatched automatic predictions are separately penalized in precision."),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="JSON output from run_specialists.sh sort")
    parser.add_argument("--truth", type=Path, required=True,
                        help="Independent box-level disposal labels, see docs/sorting-evaluation.md")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing output")
    result = evaluate(json.loads(args.predictions.read_text()),
                      json.loads(args.truth.read_text()), args.iou)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
