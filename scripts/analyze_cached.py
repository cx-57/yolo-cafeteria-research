"""Analyze cached YOLO 640px/crop/960px predictions without rerunning inference.

Uses the same 55-image cohort and threshold-specific GT matches. Incompletely
labeled images mean unmatched detections are review candidates, NOT verified FPs.
No training, relabeling, new packages, or changes to the original images.
"""
import argparse
import hashlib
import json
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


def matched(targets, predictions, confidence=0.25):
    """Replicate the original experiment's descending-confidence matching."""
    found = set()
    for p in sorted(predictions, key=lambda p: -p["conf"]):
        if p["conf"] < confidence:
            continue
        choices = [(iou(p["box"], t["box"]), idx)
                   for idx, t in enumerate(targets)
                   if idx not in found and p["class"] == t["class"]]
        if choices:
            strength, idx = max(choices)
            if strength >= 0.5:
                found.add(idx)
    return found


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
    for row in rows:
        predictions = detections(row, method, crop_conf, nms_iou)
        found = matched(row["targets"], predictions)
        source = row["image"].split("__", 1)[0]
        if source not in sources:
            sources[source] = Counter()
        for bucket in (totals, sources[source]):
            bucket["images"] += 1
            bucket["predictions"] += len(predictions)
            bucket["matched"] += len(found)
            bucket["targets"] += len(row["targets"])
        for i, target in enumerate(row["targets"]):
            stats = classes[CLASSES[target["class"]]]
            stats["targets"] += 1
            stats["matched"] += (i in found)
            if target["area"] < 0.01:
                stats["tiny_targets"] += 1
                stats["tiny_matched"] += (i in found)
            if i not in found:
                review.append({"image": row["image"], "class": CLASSES[target["class"]],
                               "area_fraction": target["area"], "issue": "missed_gt"})
    def summary(counter):
        return {**dict(counter),
                "recall": (round(counter["matched"] / counter["targets"], 4)
                           if counter["targets"] else None),
                "unmatched_predictions": counter["predictions"] - counter["matched"]}
    return {"method": method, "crop_conf": crop_conf, "nms_iou": nms_iou,
            "overall": summary(totals),
            "classes": {name: summary(counts) for name, counts in classes.items()},
            "sources": {name: summary(counts) for name, counts in sources.items()},
            "missed_gt": review}


def partition(rows, seed=42):
    """A fixed development/holdout split; diagnostic only, not independent test."""
    development, holdout = [], []
    for row in rows:
        image = row["image"].split(".rf.")[0]
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
    # The tradeoff budget is user-readable, not a guarantee of verified precision.
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("model/runs/crop-diagnostic"))
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    output = args.output or args.input / "cached_analysis.json"
    base = read_cache(args.input / "progress.jsonl")
    high = read_cache(args.input / "resolution_960_progress.jsonl")
    report = build_report(join_caches(base, high))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    selected = report["selected_on_development_only"]
    print(f"Analyzed {report['images']} cached images; {report['development_images']} development, "
          f"{report['diagnostic_holdout_images']} diagnostic holdout")
    print("Selected for analysis:", selected)
    for item in report["candidates"]:
        if {k: item[k] for k in selected} == selected:
            held = item["holdout"]["overall"]
            print(f"Held-out proxy: {held['matched']}/{held['targets']} targets matched, "
                  f"{held['unmatched_predictions']} unmatched predictions")
    print("Saved:", output)


if __name__ == "__main__":
    main()
