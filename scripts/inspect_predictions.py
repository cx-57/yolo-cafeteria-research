"""Render actual YOLO detections beside current labels; no training or relabeling.

Use the saved 11-class best.pt. Reports at one confidence/IoU threshold are
diagnostics, NOT AP/mAP and NOT independent NHS test performance.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw
import yaml

NAMES = ("food", "plate", "paper_boat", "utensil", "hinged_container",
         "napkin", "drink_carton", "condiment_packet", "wrapper", "bottle", "bowl")
EXT = {".jpg", ".jpeg", ".png", ".webp"}


def box_iou(a, b):
    x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    shared = x*y
    return shared/(area_a+area_b-shared) if area_a+area_b>shared else 0.0


def area_fraction(box, width, height):
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions")
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1]) / (width * height)


def size_band(fraction):
    if fraction < 0.01:
        return "tiny_under_1pct"
    if fraction < 0.04:
        return "small_1_to_4pct"
    if fraction < 0.16:
        return "medium_4_to_16pct"
    return "large_16pct_plus"


def sample_across_sizes(matches, cls, count):
    """Inspect tiny and big examples as well as the middle of the size spectrum."""
    ranked = sorted(matches, key=lambda entry: min(
        area_fraction(target[1], *entry[2]) for target in entry[1] if target[0] == cls))
    if len(ranked) <= count:
        return ranked
    if count == 1:
        return [ranked[0]]
    indices = [round(index * (len(ranked) - 1) / (count - 1)) for index in range(count)]
    return [ranked[index] for index in indices]


def parse_labels(path, width, height):
    if not path.exists():
        raise ValueError(f"Missing label file: {path}")
    output = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO row in {path}")
        cls = int(parts[0])
        x, y, w, h = map(float, parts[1:])
        if not 0 <= cls < len(NAMES) or w <= 0 or h <= 0:
            raise ValueError(f"Invalid label class or area in {path}")
        output.append((cls, ((x-w/2)*width, (y-h/2)*height,
                             (x+w/2)*width, (y+h/2)*height), None))
    return output


def match(targets, predictions, min_iou=0.5):
    """Maximum-cardinality one-to-one same-class matches at fixed IoU.

    Highest-IoU-first greedy matching can undercount stacked/overlapping
    plates. Augmenting paths preserve the maximum number of valid matches.
    """
    neighbors = []
    for target in targets:
        options = [(box_iou(target[1], prediction[1]), j)
                   for j, prediction in enumerate(predictions)
                   if target[0] == prediction[0]]
        neighbors.append([j for overlap, j in sorted(options, reverse=True)
                          if overlap >= min_iou])
    assigned = {}  # prediction index -> target index

    def augment(i, visited):
        for j in neighbors[i]:
            if j in visited:
                continue
            visited.add(j)
            if j not in assigned or augment(assigned[j], visited):
                assigned[j] = i
                return True
        return False

    for i in range(len(targets)):
        augment(i, set())
    return set(assigned.values()), set(assigned)


def wrong_class_candidates(targets, predictions, matched_gt, matched_pred, min_iou):
    """Rank *possible* class confusions; overlapping categories can coexist."""
    candidates = []
    for i, target in enumerate(targets):
        if i in matched_gt:
            continue
        for j, prediction in enumerate(predictions):
            if j in matched_pred or target[0] == prediction[0]:
                continue
            overlap = box_iou(target[1], prediction[1])
            if overlap >= min_iou:
                candidates.append((overlap, i, j))
    # Each GT and prediction appears at most once in the review queue.
    chosen, gt_used, pred_used = [], set(), set()
    for overlap, i, j in sorted(candidates, reverse=True):
        if i not in gt_used and j not in pred_used:
            chosen.append((i, j, overlap))
            gt_used.add(i)
            pred_used.add(j)
    return chosen


def draw_boxes(image, objects, rgb):
    picture = image.copy().convert("RGB")
    painter = ImageDraw.Draw(picture)
    for cls, coords, score in objects:
        painter.rectangle(tuple(coords), outline=rgb, width=3)
        caption = NAMES[cls] + (f" {score:.2f}" if score is not None else "")
        painter.text((max(0, coords[0]), max(0, coords[1]-13)), caption,
                     fill=rgb, stroke_width=1, stroke_fill="black")
    return picture


def inspect(data_path, weights, output, split, wanted, per_class, conf, iou, device):
    from ultralytics import YOLO

    data_path = Path(data_path).resolve()
    metadata = yaml.safe_load(data_path.read_text())
    names = metadata["names"]
    if isinstance(names, dict):
        names = [names.get(i, names.get(str(i))) for i in range(len(NAMES))]
    if list(names) != list(NAMES):
        raise ValueError("Expected the 11-class cafeteria data.yaml")
    model = YOLO(str(Path(weights).resolve()))
    names = model.names
    if isinstance(names, dict):
        names = [names.get(i, names.get(str(i))) for i in range(len(NAMES))]
    if list(names) != list(NAMES):
        raise ValueError("Checkpoint does not use the 11-class cafeteria ontology")
    root = Path(metadata.get("path") or data_path.parent)
    if not root.is_absolute():
        root = data_path.parent / root
    images_dir = Path(metadata[split])
    if not images_dir.is_absolute():
        images_dir = root / images_dir
    label_dir = images_dir.parent.parent / "labels" / split
    candidates = []
    for image in sorted(images_dir.iterdir()):
        if image.suffix.lower() not in EXT:
            continue
        with Image.open(image) as picture:
            targets = parse_labels(label_dir/(image.stem+".txt"), *picture.size)
            dimensions = picture.size
        candidates.append((image, targets, dimensions))
    picks, seen, support = [], set(), {}
    for name in wanted:
        cls = NAMES.index(name)
        matches = [(image, targets, dimensions) for image, targets, dimensions in candidates
                   if any(target[0] == cls for target in targets)]
        support[name] = len(matches)
        for image, targets, dimensions in sample_across_sizes(matches, cls, per_class):
            if image.name not in seen:
                picks.append((image, targets, dimensions))
                seen.add(image.name)
    output.mkdir(parents=True, exist_ok=True)
    report = []
    counts = defaultdict(Counter)
    area_counts = defaultdict(Counter)
    review_queue = []
    confusion_queue = []
    for image, targets, dimensions in picks:
        prediction = model.predict(str(image), imgsz=640, conf=conf, device=device,
                                   verbose=False)[0]
        boxes = [(int(b.cls.item()), tuple(float(v) for v in b.xyxy[0]),
                  float(b.conf.item())) for b in prediction.boxes]
        matched_gt, matched_pred = match(targets, boxes, iou)
        for gi, pj, overlap in wrong_class_candidates(
                targets, boxes, matched_gt, matched_pred, iou):
            gt_class, gt_box, _ = targets[gi]
            pred_class, pred_box, pred_conf = boxes[pj]
            confusion_queue.append({
                "image": image.name, "ground_truth_class": NAMES[gt_class],
                "predicted_class": NAMES[pred_class],
                "iou": round(overlap, 5), "prediction_confidence": round(pred_conf, 5),
                "gt_box_xyxy": json.dumps(gt_box),
                "prediction_box_xyxy": json.dumps(pred_box),
            })
        width, height = dimensions
        for index, (cls, bounds, _) in enumerate(targets):
            name = NAMES[cls]
            band = size_band(area_fraction(bounds, width, height))
            counts[name]["targets"] += 1
            area_counts[(name, band)]["targets"] += 1
            if index in matched_gt:
                counts[name]["matched"] += 1
                area_counts[(name, band)]["matched"] += 1
            else:
                review_queue.append({
                    "image": image.name, "class": name, "issue": "unmatched_label",
                    "area_fraction": round(area_fraction(bounds, width, height), 6),
                    "size_band": band, "confidence": "", "box_xyxy": json.dumps(bounds),
                })
        for index, (cls, bounds, confidence) in enumerate(boxes):
            if index not in matched_pred:
                name = NAMES[cls]
                counts[name]["unmatched_predictions"] += 1
                review_queue.append({
                    "image": image.name, "class": name, "issue": "unmatched_prediction",
                    "area_fraction": round(area_fraction(bounds, width, height), 6),
                    "size_band": size_band(area_fraction(bounds, width, height)),
                    "confidence": round(confidence, 5), "box_xyxy": json.dumps(bounds),
                })
        with Image.open(image) as original:
            left = draw_boxes(original, targets, "lime")
            right = draw_boxes(original, boxes, "red")
        montage = Image.new("RGB", (left.width+right.width, left.height))
        montage.paste(left, (0, 0))
        montage.paste(right, (left.width, 0))
        montage.save(output/(image.stem+".jpg"), quality=85)
        report.append({"image": image.name, "gt": len(targets), "pred": len(boxes),
                       "matched": len(matched_gt), "unmatched_targets": len(targets)-len(matched_gt),
                       "unmatched_predictions": len(boxes)-len(matched_pred)})
    with (output/"review_queue.csv").open("w", newline="") as f:
        fields = ("image", "class", "issue", "area_fraction", "size_band", "confidence", "box_xyxy")
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(review_queue, key=lambda row: (
            row["issue"] != "unmatched_label", row["area_fraction"], row["image"])))
    with (output/"wrong_class_candidates.csv").open("w", newline="") as f:
        fields = ("image", "ground_truth_class", "predicted_class", "iou",
                  "prediction_confidence", "gt_box_xyxy", "prediction_box_xyxy")
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(confusion_queue, key=lambda row: -row["iou"]))
    (output/"summary.json").write_text(json.dumps({
        "split": split, "confidence": conf, "iou": iou,
        "images_reviewed": len(picks), "eligible_images_by_class": support,
        "wrong_class_candidates": len(confusion_queue),
        "wrong_class_warning": ("Candidates are overlapping mismatched-class boxes, "
                                "not adjudicated errors; full-scene review is necessary."),
        "reviewed": report,
        "per_class": {name: {
            "target_count": counter["targets"],
            "matched": counter["matched"],
            "unmatched_targets": counter["targets"] - counter["matched"],
            "unmatched_predictions": counter["unmatched_predictions"],
            "recall_at_threshold": (counter["matched"] / counter["targets"]
                                    if counter["targets"] else None)
        } for name, counter in sorted(counts.items())},
        "by_object_area": {name: {band: {
            "targets": counter["targets"], "matched": counter["matched"],
            "missed": counter["targets"] - counter["matched"]
        } for (category, band), counter in area_counts.items() if category == name}
            for name in sorted(counts)},
        "size_band_definitions": "Box area divided by entire image area: under 1%; 1-4%; 4-16%; 16%+.",
        "warning": ("Unmatched predictions may be missing annotations, not false positives. "
                    "Fixed-threshold matches are NOT mAP or independent NHS test accuracy. "
                    "Small-box status is NOT proof of occlusion.")
    }, indent=2)+"\n")
    print(f"Saved {len(picks)} side-by-side prediction overlays to {output}")
    print("Eligible target images:", support)
    print("Possible wrong-class overlaps:", len(confusion_queue),
          "(see wrong_class_candidates.csv)")
    if split == "train":
        print("TRAIN predictions are illustrations ONLY; do not use for test performance.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default="data/training/data.yaml")
    p.add_argument("--output", default="model/prediction-review")
    p.add_argument("--split", choices=("train", "val"), default="val")
    p.add_argument("--classes", nargs="+", choices=NAMES,
                   default=["plate", "wrapper", "bottle", "napkin"])
    p.add_argument("--per-class", type=int, default=4)
    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--iou", type=float, default=0.50)
    p.add_argument("--device")
    a = p.parse_args()
    if a.per_class < 1 or not 0 < a.conf <= 1 or not 0 < a.iou <= 1:
        p.error("Invalid limit or confidence/IoU threshold")
    inspect(a.data, a.weights, Path(a.output), a.split, a.classes,
            a.per_class, a.conf, a.iou, a.device)


if __name__ == "__main__":
    main()
