"""Resumable full-frame vs crop-inference diagnostic for the existing YOLO checkpoint.

No training or data modification. Evaluates detection recall on labeled validation
images containing plates, wrappers, or bottles. Run from the project root.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

CLASSES = {1: "plate", 8: "wrapper", 9: "bottle"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
THRESHOLDS = (0.05, 0.25)


def overlap(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / (area_a + area_b - intersection) if area_a + area_b > intersection else 0


def read_truth(label, width, height):
    targets = []
    for row in label.read_text().splitlines():
        if not row.strip():
            continue
        cls, x, y, w, h = map(float, row.split()[:5])
        if int(cls) not in CLASSES:
            continue
        x1, x2 = (x - w / 2) * width, (x + w / 2) * width
        y1, y2 = (y - h / 2) * height, (y + h / 2) * height
        targets.append({"class": int(cls), "box": [x1, y1, x2, y2], "area": w * h})
    return targets


def predictions(model, image, imgsz, device, ox=0, oy=0):
    result = model.predict(image, imgsz=imgsz, conf=min(THRESHOLDS),
                           device=device, verbose=False)[0]
    boxes = result.boxes
    if boxes is None:
        return []
    return [{"class": int(cls), "conf": float(score),
             "box": [float(box[0]) + ox, float(box[1]) + oy,
                     float(box[2]) + ox, float(box[3]) + oy]}
            for cls, score, box in zip(boxes.cls.tolist(), boxes.conf.tolist(),
                                       boxes.xyxy.tolist()) if int(cls) in CLASSES]


def crop_rectangles(width, height, fraction):
    cw, ch = max(1, round(width * fraction)), max(1, round(height * fraction))
    return [(x, y, x + cw, y + ch)
            for y in sorted(set((0, height - ch)))
            for x in sorted(set((0, width - cw)))]


def matched_targets(targets, detections, confidence):
    """One prediction can recall only one same-class ground-truth object."""
    matched = set()
    for prediction in sorted(detections, key=lambda p: -p["conf"]):
        if prediction["conf"] < confidence:
            continue
        choices = [(overlap(t["box"], prediction["box"]), i)
                   for i, t in enumerate(targets)
                   if i not in matched and t["class"] == prediction["class"]]
        if choices:
            score, index = max(choices)
            if score >= 0.5:
                matched.add(index)
    return matched


def summarize(rows):
    output = {}
    for conf in THRESHOLDS:
        section = {}
        for mode in ("full", "crops", "combined"):
            values = {}
            for cls, name in CLASSES.items():
                total = found = small_total = small_found = 0
                for row in rows:
                    matched = matched_targets(row["targets"], row[mode], conf)
                    for i, target in enumerate(row["targets"]):
                        if target["class"] != cls:
                            continue
                        total += 1
                        found += i in matched
                        if target["area"] < 0.01:
                            small_total += 1
                            small_found += i in matched
                values[name] = {
                    "recall": round(found / total, 4) if total else None,
                    "matched": found, "total": total,
                    "small_under_1pct_matched": small_found,
                    "small_under_1pct_total": small_total,
                }
            section[mode] = values
        output[str(conf)] = section
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path(
        "model/runs/yolo11n-corrected-baseline-100/weights/best.pt"))
    parser.add_argument("--dataset", type=Path, default=Path("data/training"))
    parser.add_argument("--output", type=Path, default=Path("model/runs/crop-diagnostic"))
    parser.add_argument("--max-images", type=int, default=55)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--crop-fraction", type=float, default=0.65)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"Checkpoint not found: {args.weights}")
    if not 0.5 <= args.crop_fraction < 1 or args.max_images < 1:
        parser.error("Use 0.5 <= --crop-fraction < 1 and --max-images >= 1")

    folder = args.dataset / "images/val"
    eligible = []
    for image in sorted(folder.iterdir()):
        if image.suffix.lower() not in IMAGE_EXTS:
            continue
        label = args.dataset / "labels/val" / (image.stem + ".txt")
        if not label.is_file():
            continue
        with Image.open(image) as im:
            truth = read_truth(label, *im.size)
        if truth:
            eligible.append((image, {t["class"] for t in truth}))
    # Round-robin avoids selecting only one class from alphabetically sorted files.
    images = []
    while len(images) < args.max_images:
        progress = False
        for cls in CLASSES:
            for image, classes in eligible:
                if cls in classes and image not in images:
                    images.append(image)
                    progress = True
                    break
            if len(images) >= args.max_images:
                break
        if not progress:
            break
    if not images:
        parser.error("No labeled validation images with plates, wrappers, or bottles")

    # Distinct cache keys prevent reusing results from other checkpoints/settings.
    digest = hashlib.sha256(args.weights.read_bytes()).hexdigest()[:16]
    settings = {"weights_sha256_prefix": digest, "imgsz": args.imgsz,
                "crop_fraction": args.crop_fraction, "device": args.device}
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "progress.jsonl"
    def input_signature(image):
        label = args.dataset / "labels/val" / (image.stem + ".txt")
        stat = image.stat()
        return [stat.st_size, stat.st_mtime_ns,
                hashlib.sha256(label.read_bytes()).hexdigest()[:16]]

    done = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            try:
                row = json.loads(line)
                image = folder / row["image"]
                if (row.get("settings") == settings and image in images
                        and row.get("input_signature") == input_signature(image)):
                    done[row["image"]] = row
            except (ValueError, KeyError):
                continue

    remaining = [img for img in images if img.name not in done]
    if remaining:
        model = YOLO(str(args.weights))
        with cache.open("a") as f:
            for i, image in enumerate(remaining, 1):
                with Image.open(image) as source:
                    im = source.convert("RGB")
                truth = read_truth(args.dataset / "labels/val" / (image.stem + ".txt"),
                                   *im.size)
                full = predictions(model, im, args.imgsz, args.device)
                crops = []
                for rect in crop_rectangles(*im.size, args.crop_fraction):
                    x1, y1, x2, y2 = rect
                    crops.extend(predictions(model, im.crop(rect), args.imgsz,
                                             args.device, x1, y1))
                row = {"settings": settings, "image": image.name,
                       "input_signature": input_signature(image), "targets": truth,
                       "full": full, "crops": crops, "combined": full + crops}
                f.write(json.dumps(row) + "\n")
                f.flush()  # A timeout preserves all previously finished images.
                done[image.name] = row
                print(f"{len(done)}/{len(images)}: {image.name}", flush=True)
    else:
        print(f"Resumed all {len(images)} images from {cache}")

    result = {"settings": settings, "images": len(images),
              "comparison": summarize([done[image.name] for image in images]),
              "interpretation": "Validation recall at IoU>=0.50 only. Incomplete labels prevent reliable FP/precision claims; cropping can increase duplicate predictions."}
    recovered = []
    for image in images:
        row = done[image.name]
        full_hits = matched_targets(row["targets"], row["full"], 0.25)
        combined_hits = matched_targets(row["targets"], row["combined"], 0.25)
        for index in sorted(combined_hits - full_hits):
            target = row["targets"][index]
            recovered.append({"image": image.name, "class": CLASSES[target["class"]],
                              "area_fraction": round(target["area"], 6)})
    (args.output / "recovered_examples.json").write_text(
        json.dumps(recovered, indent=2) + "\n")
    path = args.output / "summary.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["comparison"], indent=2))
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
