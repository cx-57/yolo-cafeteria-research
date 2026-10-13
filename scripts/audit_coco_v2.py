"""Independent COCO-detector cross-check of every final V2 image.

All 3,096 image/label pairs are visited. Flags need *visual* adjudication;
the pretrained COCO detector is not an annotator and cannot check non-COCO
classes (plate, wrapper, napkin, paper boat, condiment packet, carton).
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import yaml

# COCO class IDs mapped to the cafeteria 11-class ontology.
COCO_TO_NHS = {
    39: 9,                         # bottle
    42: 3, 43: 3, 44: 3,         # fork / knife / spoon -> utensil
    45: 10,                       # bowl
    46: 0, 47: 0, 48: 0, 49: 0,  # banana, apple, sandwich, orange
    50: 0, 51: 0, 52: 0, 53: 0, 54: 0, 55: 0,
}
NAMES = ("food", "plate", "paper_boat", "utensil", "hinged_container",
         "napkin", "drink_carton", "condiment_packet", "wrapper", "bottle", "bowl")
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def overlap(first, second):
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    shared = max(x2 - x1, 0) * max(y2 - y1, 0)
    a = max(first[2] - first[0], 0) * max(first[3] - first[1], 0)
    b = max(second[2] - second[0], 0) * max(second[3] - second[1], 0)
    return shared / (a + b - shared) if a + b > shared else 0.0


def original_boxes(label, width, height):
    if not label.is_file():
        raise ValueError("Missing annotated label file: " + str(label))
    rows = []
    for lineno, line in enumerate(label.read_text().splitlines(), 1):
        if not line.strip():
            continue
        words = line.split()
        if len(words) != 5:
            raise ValueError(f"{label}:{lineno}: Expected YOLO class xywh")
        klass = int(words[0])
        x, y, w, h = map(float, words[1:])
        if klass not in range(len(NAMES)):
            raise ValueError("Invalid target class: " + str(label))
        rows.append((klass, ((x-w/2)*width, (y-h/2)*height,
                             (x+w/2)*width, (y+h/2)*height)))
    return rows


def find_discrepancies(rows, detections, food_iou=0.20, other_iou=0.35):
    """A disagreement is a QUESTION to review, never an automatic correction."""
    candidates = []
    for klass, source_name, confidence, xyxy in detections:
        target = COCO_TO_NHS.get(klass)
        if target is None:
            continue
        threshold = food_iou if target == 0 else other_iou
        best = max((overlap(xyxy, bounds) for cls, bounds in rows if cls == target),
                   default=0)
        if best < threshold:
            candidates.append((target, klass, source_name, confidence, best, xyxy))
    return candidates


def run(dataset, output, conf=0.35, imgsz=640):
    from ultralytics import YOLO

    config = yaml.safe_load(Path(dataset).read_text())
    names = config["names"]
    if isinstance(names, dict):
        names = [names.get(i, names.get(str(i))) for i in range(11)]
    if list(names) != list(NAMES):
        raise ValueError("Unexpected cafeteria class order")
    root = Path(config.get("path") or Path(dataset).parent)
    if not root.is_absolute():
        root = Path(dataset).parent / root

    pairs = []
    for split in ("train", "val"):
        directory = Path(config[split])
        if not directory.is_absolute():
            directory = root / directory
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() in SUFFIXES:
                label = directory.parent.parent / "labels" / split / (path.stem + ".txt")
                pairs.append((path.resolve(), label.resolve(), split))
    if len(pairs) != 3096:
        raise ValueError(f"Expected 3,096 original V2 images, received {len(pairs)}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolo11n.pt")  # independent COCO-pretrained model, NOT project-trained
    checked, found, candidate_rows = 0, Counter(), []
    per_image = []
    images = [str(path) for path, _lab, _split in pairs]
    results = model.predict(source=images, batch=8, device="cpu", imgsz=imgsz,
                            conf=conf, verbose=False, stream=True)
    for (image, label, split), result in zip(pairs, results):
        if Path(result.path).resolve() != image:
            raise ValueError("Inference order does not match input photo paths")
        height, width = result.orig_shape
        true_boxes = original_boxes(label, width, height)
        detections = []
        for item in result.boxes:
            cls = int(item.cls.item())
            if cls not in COCO_TO_NHS:
                continue
            detections.append((cls, model.names[cls], float(item.conf.item()),
                               tuple(float(v) for v in item.xyxy[0])))
        candidates = find_discrepancies(true_boxes, detections)
        checked += 1
        found.update(NAMES[x[0]] for x in candidates)
        per_image.append({"image": image.name, "split": split,
                          "source": image.stem.split("__",1)[0],
                          "labeled_objects": len(true_boxes),
                          "independent_predictions": len(detections),
                          "candidates": len(candidates),
                          "status": "automated_candidates_not_visually_certified"})
        for target, coco, cname, conf_value, best, bounds in candidates:
            candidate_rows.append({
                "image": image.name, "split": split,
                "source": image.stem.split("__",1)[0],
                "suspected_class": NAMES[target], "coco_class": cname,
                "confidence": round(conf_value, 5), "max_existing_iou": round(best, 5),
                "predicted_box_xyxy": json.dumps([round(v,1) for v in bounds]),
                "status": "human_confirmation_required",
            })
    if checked != len(pairs):
        raise ValueError(f"Inference unexpectedly stopped at {checked} images")
    for filename, rows, keys in (
        ("candidates.csv", candidate_rows,
         ("image","split","source","suspected_class","coco_class",
          "confidence","max_existing_iou","predicted_box_xyxy","status")),
        ("per_image.csv", per_image,
         ("image","split","source","labeled_objects",
          "independent_predictions","candidates","status"))):
        with (output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "images_checked": checked, "candidate_discrepancies": len(candidate_rows),
        "candidate_classes": dict(found),
        "model": "COCO pretrained YOLO11n", "confidence": conf, "imgsz": imgsz,
        "warning": ("Only COCO-covered classes cross-checked. This is not a full "
                    "11-class semantic audit or a claim of corrected annotations. "
                    "Never automatically overwrite human-reviewed labels."),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("COCO independent label cross-check:", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.data, args.output)
