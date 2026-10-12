"""Analyze cached YOLO 640px/crop/960px predictions without rerunning inference.

Uses the same 55-image cohort and threshold-specific GT matches. Incompletely
labeled images mean unmatched detections are review candidates, NOT verified FPs.
No training, relabeling, new packages, or changes to the original images.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

CLASSES = {1: "plate", 8: "wrapper", 9: "bottle"}


def iou(a, b):
    x = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    aa = max(0., a[2]-a[0]) * max(0., a[3]-a[1])
    bb = max(0., b[2]-b[0]) * max(0., b[3]-b[1])
    intersection = x * y
    return intersection / (aa + bb - intersection) if aa + bb > intersection else 0.


def nms(predictions, threshold=0.5):
    kept = []
    for p in sorted(predictions, key=lambda p: -p["conf"]):
        if all(p["class"] != k["class"] or iou(p["box"], k["box"]) < threshold
               for k in kept):
            kept.append(p)
    return kept


def assignments(targets, predictions, confidence=0.25):
    """Maximum-cardinality same-class IoU>=0.50 matching.

    Greedy matching can undercount overlapping plates: the highest-confidence
    prediction may consume the only label a second prediction can match.
    """
    edges = {}
    for j, prediction in enumerate(predictions):
        if prediction["conf"] < confidence:
            continue
        edges[j] = [i for score, i in sorted(
            ((iou(prediction["box"], target["box"]), i)
             for i, target in enumerate(targets)
             if target["class"] == prediction["class"]), reverse=True)
            if score >= 0.5]
    by_target = {}

    def augment(j, visited):
        for i in edges[j]:
            if i in visited:
                continue
            visited.add(i)
            previous = by_target.get(i)
            if previous is None or augment(previous, visited):
                by_target[i] = j
                return True
        return False

    for j in sorted(edges, key=lambda j: -predictions[j]["conf"]):
        augment(j, set())
    return by_target


def matched(targets, predictions, confidence=0.25):
    return set(assignments(targets, predictions, confidence))


def source_group(filename):
    """Group old split names and alternate Roboflow exports of one photograph."""
    name = filename.split(".rf.")[0]
    if "__" not in name:
        return name
    source, original = name.split("__", 1)
    original = re.sub(r"^(train|valid|val|test)__", "", original)
    if source == "tartanhacks":
        video_id = re.search(r"IMG_\d+", original)
        if video_id:
            original = video_id.group()
    return source + "__" + original


def read_cache(path):
    rows = {}
    with path.open() as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("image"), str):
                raise ValueError(f"Invalid image entry in {path}")
            rows[row["image"]] = row
    if not rows:
        raise ValueError(f"Empty cache: {path}")
    return rows


def join_caches(base, high):
    if set(base) != set(high):
        raise ValueError("The 640px and 960px caches do not cover identical images")
    rows = []
    for image in sorted(base):
        a, b = base[image], high[image]
        settings_a, settings_b = a.get("settings", {}), b.get("settings", {})
        if (a.get("input_signature") != b.get("input_signature")
                or settings_a.get("weights_sha256_prefix") != settings_b.get("weights_sha256_prefix")
                or settings_a.get("imgsz") != 640
                or settings_b.get("compare_imgsz") != 960
                or settings_a.get("device") != settings_b.get("device")
                or not isinstance(a.get("targets"), list)):
            raise ValueError(f"Inconsistent checkpoint, image/labels, or settings: {image}")
        rows.append({"image": image, "targets": a["targets"],
                     "full": a["full"], "crops": a["crops"], "high": b["full"]})
    return rows


def candidates():
    for threshold in (0.1, 0.15, 0.2, 0.25, 0.35):
        yield ("full640", threshold, None)
        yield ("full960", threshold, None)
    for threshold in (0.25, 0.35, 0.45, 0.55, 0.7):
        for nms_iou in (0.3, 0.5, 0.7):
            yield ("640_plus_crops", threshold, nms_iou)


def detections(row, method, crop_conf=None, nms_iou=None):
    if method == "full640":
        return [p for p in row["full"] if p["conf"] >= (crop_conf if crop_conf is not None else 0.25)]
    if method == "full960":
        return [p for p in row["high"] if p["conf"] >= (crop_conf if crop_conf is not None else 0.25)]
    full = [p for p in row["full"] if p["conf"] >= 0.25]
    crops = [p for p in row["crops"] if p["conf"] >= crop_conf]
    return nms(full + crops, nms_iou)


def evaluate(rows, method, crop_conf=None, nms_iou=None):
    totals = Counter()
    classes = {name: Counter() for name in CLASSES.values()}
    sources = {}
    review = []
    unmatched_review = []
    sizes = {name: Counter() for name in
             ("tiny_under_1pct", "small_1_to_4pct", "medium_4_to_16pct", "large_16pct_plus")}
    for row in rows:
        predictions = detections(row, method, crop_conf, nms_iou)
        assignments_ = assignments(row["targets"], predictions)
        found = set(assignments_)
        used_predictions = set(assignments_.values())
        source = row["image"].split("__", 1)[0]
        if source not in sources:
            sources[source] = Counter()
        for bucket in (totals, sources[source]):
            bucket["images"] += 1
            bucket["predictions"] += len(predictions)
            bucket["matched"] += len(found)
            bucket["targets"] += len(row["targets"])
        for j, prediction in enumerate(predictions):
            classes[CLASSES[prediction["class"]]]["predictions"] += 1
            if j in used_predictions:
                continue
            best_overlap = max((iou(prediction["box"], t["box"])
                                for t in row["targets"]), default=0.)
            best_other = max((iou(prediction["box"], t["box"])
                              for t in row["targets"]
                              if t["class"] != prediction["class"]), default=0.)
            unmatched_review.append({
                "image": row["image"], "class": CLASSES[prediction["class"]],
                "confidence": round(prediction["conf"], 5), "xyxy": prediction["box"],
                "max_iou_any_label": round(best_overlap, 4),
                "max_iou_other_class": round(best_other, 4),
                "review_hint": ("potential_class_confusion" if best_other >= 0.5
                                else "review_unlabeled_or_duplicate_or_wrong"),
            })
        for i, target in enumerate(row["targets"]):
            stats = classes[CLASSES[target["class"]]]
            area = target["area"]
            size = ("tiny_under_1pct" if area < .01 else
                    "small_1_to_4pct" if area < .04 else
                    "medium_4_to_16pct" if area < .16 else "large_16pct_plus")
            sizes[size]["targets"] += 1
            sizes[size]["matched"] += (i in found)
            stats["targets"] += 1
            stats["matched"] += (i in found)
            if area < .01:
                stats["tiny_targets"] += 1
                stats["tiny_matched"] += (i in found)
            if i not in found:
                review.append({"image": row["image"], "class": CLASSES[target["class"]],
                               "area_fraction": area, "issue": "missed_gt"})

    def summary(counter):
        result = {**dict(counter),
                  "recall": (round(counter["matched"] / counter["targets"], 4)
                             if counter["targets"] else None)}
        if "predictions" in counter:
            result["unmatched_predictions"] = counter["predictions"] - counter["matched"]
        return result

    return {"method": method, "crop_conf": crop_conf, "nms_iou": nms_iou,
            "overall": summary(totals),
            "classes": {name: summary(counts) for name, counts in classes.items()},
            "sources": {name: summary(counts) for name, counts in sources.items()},
            "sizes": {name: summary(counts) for name, counts in sizes.items()},
            "missed_gt": review, "unmatched_review": unmatched_review}


def partition(rows, seed=42):
    """A fixed development/holdout split; diagnostic only, not independent test."""
    development, holdout = [], []
    for row in rows:
        image = source_group(row["image"])
        value = int.from_bytes(hashlib.sha256(f"{seed}:{image}".encode()).digest()[:8], "big")
        (holdout if value % 10 < 3 else development).append(row)
    if not development or not holdout:
        raise ValueError("Diagnostic split is empty; use more validation images")
    return development, holdout


def build_report(rows):
    development, holdout = partition(rows)
    evaluated = []
    for mode, conf, nms_iou in candidates():
        dev = evaluate(development, mode, conf, nms_iou)
        test = evaluate(holdout, mode, conf, nms_iou)
        all_images = evaluate(rows, mode, conf, nms_iou)
        evaluated.append({"method": mode, "crop_conf": conf, "nms_iou": nms_iou,
                          "dev": dev, "holdout": test, "all": all_images})
    baseline = next(e for e in evaluated if e["method"] == "full640" and e["crop_conf"] == 0.25)["dev"]["overall"]
    # The unmatched-prediction budget is a review-load proxy, NOT a precision guarantee.
    def key(entry):
        stats = entry["dev"]["overall"]
        return (-stats["matched"], stats["unmatched_predictions"], stats["predictions"])
    limited = [e for e in evaluated if e["dev"]["overall"]["unmatched_predictions"]
               <= baseline["unmatched_predictions"] + 15]
    picked = min(limited or evaluated[:1], key=key)
    return {"images": len(rows), "development_images": len(development),
            "diagnostic_holdout_images": len(holdout),
            "selected_on_development_only": {k: picked[k] for k in ("method", "crop_conf", "nms_iou")},
            "selection_budget": "At most 15 additional unmatched predictions in development images",
            "candidates": evaluated,
            "limitations": [
                "Holdout here is drawn from the SAME contaminated validation corpus, not an NHS test set.",
                "Annotation incompleteness means unmatched predictions are not confirmed false positives.",
                "Threshold selection on development data can overfit; do not treat tuning as a model improvement.",
                "Inference cost for full640/full960/crops differs; latency was not measured.",
            ]}



def render_review_overlays(rows, images_dir, output, limit=12):
    """Draw side-by-side examples; never modify images or training labels."""
    from PIL import Image, ImageDraw
    if limit < 1:
        return 0
    output.mkdir(parents=True, exist_ok=True)

    def priority(row):
        found = matched(row["targets"], row["full"])
        missed = len(row["targets"]) - len(found)
        tiny = sum(i not in found and t["area"] < .01
                   for i, t in enumerate(row["targets"]))
        return tiny * 2 + missed, missed, len(row["targets"])

    picks = sorted(rows, key=priority, reverse=True)[:limit]
    for row in picks:
        path = images_dir / row["image"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing image for overlay: {path}")
        with Image.open(path) as image:
            original = image.convert("RGB")
        modes = [
            ("Labels: green=matched, yellow=missed", None),
            ("Full frame 640px", row["full"]),
            ("Full frame 960px", row["high"]),
            ("640px + crops / NMS", detections(row, "640_plus_crops", 0.25, 0.5)),
        ]
        panels = []
        for title, predictions in modes:
            panel = original.copy()
            painter = ImageDraw.Draw(panel)
            if predictions is None:
                found = matched(row["targets"], row["full"])
                for i, target in enumerate(row["targets"]):
                    color = "lime" if i in found else "yellow"
                    painter.rectangle(target["box"], outline=color, width=3)
                    painter.text((target["box"][0], max(0, target["box"][1] - 14)),
                                 CLASSES[target["class"]], fill=color,
                                 stroke_fill="black", stroke_width=1)
            else:
                present = [p for p in predictions if p["conf"] >= 0.25]
                assigned = set(assignments(row["targets"], present).values())
                for i, prediction in enumerate(present):
                    color = "cyan" if i in assigned else "red"
                    painter.rectangle(prediction["box"], outline=color, width=3)
                    painter.text((prediction["box"][0], max(0, prediction["box"][1] - 14)),
                                 f"{CLASSES[prediction['class']]} {prediction['conf']:.2f}",
                                 fill=color, stroke_fill="black", stroke_width=1)
            if panel.width > 640:
                panel = panel.resize((640, round(640 * panel.height / panel.width)))
            painter = ImageDraw.Draw(panel)
            painter.rectangle((0, 0, min(panel.width, 500), 20), fill="black")
            painter.text((5, 4), title, fill="white")
            panels.append(panel)
        width = max(panel.width for panel in panels)
        height = max(panel.height for panel in panels)
        canvas = Image.new("RGB", (width * 2, height * 2), "black")
        for i, panel in enumerate(panels):
            canvas.paste(panel, ((i % 2) * width, (i // 2) * height))
        canvas.save(output / (Path(row["image"]).stem + "_comparison.jpg"), quality=88)
    return len(picks)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("model/runs/crop-diagnostic"))
    p.add_argument("--output", type=Path)
    p.add_argument("--review-overlays", type=int, default=0,
                   help="Render N hardest development images side by side")
    p.add_argument("--images-dir", type=Path, default=Path("data/training/images/val"))
    args = p.parse_args()
    output = args.output or args.input / "cached_analysis.json"
    base = read_cache(args.input / "progress.jsonl")
    high = read_cache(args.input / "resolution_960_progress.jsonl")
    report = build_report(join_caches(base, high))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    chosen = report["selected_on_development_only"]
    picked = next(item for item in report["candidates"]
                  if all(item[key] == value for key, value in chosen.items()))
    with (output.parent / "cached_review_queue.csv").open("w", newline="") as stream:
        columns = ["image", "class", "confidence", "xyxy", "max_iou_any_label",
                   "max_iou_other_class", "review_hint"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in sorted(picked["holdout"]["unmatched_review"],
                          key=lambda row: -row["confidence"]):
            writer.writerow(row)
    selected = report["selected_on_development_only"]
    print(f"Analyzed {report['images']} cached images; {report['development_images']} development, "
          f"{report['diagnostic_holdout_images']} diagnostic holdout")
    print("Selected for analysis:", selected)
    for item in report["candidates"]:
        if {k: item[k] for k in selected} == selected:
            held = item["holdout"]["overall"]
            print(f"Held-out proxy: {held['matched']}/{held['targets']} targets matched, "
                  f"{held['unmatched_predictions']} unmatched predictions")
    if args.review_overlays:
        development, _ = partition(join_caches(base, high))
        count = render_review_overlays(development, args.images_dir,
                                       output.parent / "cached_review_overlays", args.review_overlays)
        print(f"Saved {count} four-panel diagnostic JPEGs")
    print("Saved:", output)


if __name__ == "__main__":
    main()
