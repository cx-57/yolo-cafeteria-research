"""Prepare the 80/20 dataset, train YOLO11, and evaluate the model."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from PIL import Image
import yaml

from prepare import PROJECT_ROOT, IMAGE_SUFFIXES, LEGACY_SOURCES, current_label, remap_label_line

DATA = PROJECT_ROOT / "data"
OUTPUT = DATA / "training"
MODEL = PROJECT_ROOT / "model"
TRAIN_ONLY = {
    "condiments", "tartanhacks", "waste_detection_v3",
    "dry_wet_waste", "school_lunch_asian", "food_american",
}
SOURCES = LEGACY_SOURCES | TRAIN_ONLY


def class_names():
    classes = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())["classes"]
    return [classes[i] for i in range(len(classes))]


def group_for_split(source, filename):
    """Group multiple Roboflow exports of the same original photo."""
    name = re.sub(r"^(train|valid|val|test)__", "", filename).split(".rf.")[0]
    if source == "tartanhacks":
        match = re.search(r"IMG_\d+", name)
        if match:
            return match.group()
    return name


def read_labels(path, source, names):
    """Validate boxes, correct tiny export-rounding errors, preserve source files."""
    valid = []
    corrected = 0
    names_by_id = dict(enumerate(names))
    ids_by_name = {name: i for i, name in enumerate(names)}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = current_label(line, source).split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{number}: expected 5 numbers")
        x, y, w, h = map(float, parts[1:])
        left, right, top, bottom = x-w/2, x+w/2, y-h/2, y+h/2
        excess = max(0, -left, right-1, -top, bottom-1)
        if 0 < excess <= 0.0001 and w > 0 and h > 0:
            left, right = max(0, left), min(1, right)
            top, bottom = max(0, top), min(1, bottom)
            parts[1:] = [f"{(left+right)/2:.9f}", f"{(top+bottom)/2:.9f}",
                         f"{right-left:.9f}", f"{bottom-top:.9f}"]
            corrected += 1
        row = remap_label_line(" ".join(parts), names_by_id,
                               ids_by_name, path, number)
        if row is None:
            raise ValueError(f"{path}:{number}: unrecognized class")
        valid.append(row)
    if not valid:
        raise ValueError(f"Missing object annotations: {path}")
    return valid, corrected


def collect_images(names):
    """Read originals; never use data/raw or unreviewed candidates for training."""
    groups, duplicates = defaultdict(list), defaultdict(set)
    sources, corrections = Counter(), Counter()
    for source in sorted(SOURCES):
        images = DATA / "processed" / source / "images"
        labels = DATA / "processed" / source / "labels"
        if not images.is_dir():
            continue
        for image in sorted(images.iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = labels / (image.stem + ".txt")
            rows, fixed = read_labels(label, source, names)
            with Image.open(image) as opened:
                rgb = opened.convert("RGB")
                pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            group = source + ":" + group_for_split(source, image.stem)
            groups[group].append((source, image, rows))
            duplicates[pixels].add(group)
            sources[source] += 1
            corrections[source] += fixed
        for label in labels.glob("*.txt"):
            if not any((images / (label.stem + ext)).exists() for ext in IMAGE_SUFFIXES):
                raise ValueError(f"Label without image: {label}")
    if not groups:
        raise ValueError("No processed photos found")

    # Exact pixel duplicates must stay in the same split, even across sources.
    parent = {key: key for key in groups}

    def root(key):
        while key != parent[key]:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for matching in duplicates.values():
        first = min(matching)
        for key in matching:
            parent[root(key)] = root(first)
    combined = defaultdict(list)
    for key, photos in groups.items():
        combined[root(key)].extend(photos)
    return combined, sources, corrections


def build_dataset(augment_factor=10, seed=42):
    if not 1 <= augment_factor <= 20:
        raise ValueError("Augmentation factor must be 1–20")
    names = class_names()
    groups, source_counts, corrections = collect_images(names)
    total = sum(source_counts.values())
    target = round(total * 0.20)

    # Pick ~20% of ALL originals for validation, but never split a photo group
    # or put known correlated training-only sources into validation.
    eligible = sorted(
        (key for key, photos in groups.items()
         if not any(source in TRAIN_ONLY for source, _, _ in photos)),
        key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).digest(),
    )
    validation = set()
    selected = 0
    for key in eligible:
        if selected + len(groups[key]) <= target:
            validation.add(key)
            selected += len(groups[key])

    if not validation:
        raise ValueError("No independent validation photos available")
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)  # Generated copies only; never changes source photos.

    counts = Counter()
    class_counts = {"train": Counter(), "val": Counter()}
    for group, photos in sorted(groups.items()):
        split = "val" if group in validation else "train"
        for source, image, rows in photos:
            stem = source + "__" + image.stem
            image_dir = OUTPUT / "images" / split
            label_dir = OUTPUT / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            output_image = image_dir / (stem + image.suffix)
            output_label = label_dir / (stem + ".txt")
            if output_image.exists() or output_label.exists():
                raise ValueError(f"Duplicate output name: {stem}")
            try:
                os.link(image, output_image)
            except OSError:
                shutil.copy2(image, output_image)
            output_label.write_text("\n".join(rows) + "\n")
            counts[split] += 1
            class_counts[split].update(set(int(row.split()[0]) for row in rows))

            # Hardlinks avoid storing 10 physical copies. Ultralytics applies
            # random, bbox-aware augmentations to each training view every epoch.
            if split == "train":
                for n in range(1, augment_factor):
                    suffix = f"__view{n:02d}"
                    os.link(output_image, image_dir / (stem + suffix + image.suffix))
                    os.link(output_label, label_dir / (stem + suffix + ".txt"))

    if counts["val"] != target:
        print(f"Grouped images prevent exact 80/20: {counts['val']} val, target {target}")
    data_yaml = OUTPUT / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(OUTPUT.resolve()), "train": "images/train",
        "val": "images/val", "names": names}, sort_keys=False))
    report = {
        "original_images": total, "train_originals": counts["train"],
        "val_originals": counts["val"], "test_originals": 0,
        "target_validation_originals": target,
        "train_views": counts["train"] * augment_factor,
        "augment_factor": augment_factor, "seed": seed,
        "source_images": dict(source_counts),
        "minor_boundary_rounding_clips_in_generated_labels": dict(corrections),
        "train_class_images_original": {name: class_counts["train"][i] for i, name in enumerate(names)},
        "val_class_images_original": {name: class_counts["val"][i] for i, name in enumerate(names)},
        "missing_train_classes": [name for i, name in enumerate(names) if not class_counts["train"][i]],
        "missing_val_classes": [name for i, name in enumerate(names) if not class_counts["val"][i]],
    }
    (OUTPUT / "split_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{total} originals: {counts['train']} train, {counts['val']} val, "
          f"{report['train_views']} training views ({augment_factor}x)")
    print("No test photos included. Collect independent NHS test photos.")
    return data_yaml


def check_data(data_file):
    data = yaml.safe_load(Path(data_file).read_text())
    names = data["names"]
    if isinstance(names, dict):
        names = [names.get(i, names.get(str(i))) for i in range(len(class_names()))]
    if names != class_names():
        raise ValueError("Wrong class names or IDs in data.yaml")
    return data


def evaluate(weights, data_file, split="val", device=None):
    from ultralytics import YOLO
    data = check_data(data_file)
    if split == "test" and not data.get("test"):
        raise ValueError("Add independent test: images/path to your data.yaml first")
    options = dict(data=str(data_file), split=split, project=str(MODEL / "runs"),
                   name="evaluation_" + split, plots=True)
    if device:
        options["device"] = device
    metrics = YOLO(str(weights)).val(**options)
    ap = {int(i): {"ap50": float(a), "map50_95": float(b)} for i, a, b
          in zip(metrics.box.ap_class_index, metrics.box.ap50, metrics.box.ap)}
    result = {
        "split": split, "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr), "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class": {name: ap.get(i) for i, name in enumerate(class_names())},
    }
    location = Path(metrics.save_dir) / "metrics.json"
    location.write_text(json.dumps(result, indent=2) + "\n")
    print("Results:", location)


def load_model(name):
    from ultralytics import YOLO
    if name != "yolo11n.pt":
        return YOLO(name)
    weights = MODEL / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    target = weights / name
    if target.exists():
        return YOLO(str(target))
    # Ultralytics downloads pretrained YOLO11n into the current directory.
    # Download into model/weights/ rather than cluttering the project root.
    original = Path.cwd()
    try:
        os.chdir(weights)
        return YOLO(name)
    finally:
        os.chdir(original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", type=Path, help="Prepared data.yaml")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--augment-factor", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", help="mps on Apple Silicon; 0 on NVIDIA")
    parser.add_argument("--name", default="yolo11n-10x")
    parser.add_argument("--resume", type=Path, metavar="LAST_PT")
    parser.add_argument("--evaluate", type=Path, metavar="BEST_PT")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()
    if args.resume:
        from ultralytics import YOLO
        YOLO(str(args.resume)).train(resume=True)
        return
    if args.evaluate:
        evaluate(args.evaluate, args.data or OUTPUT / "data.yaml", args.split, args.device)
        return
    if args.split == "test":
        parser.error("--split test requires --evaluate")
    if min(args.epochs, args.batch, args.imgsz) < 1:
        parser.error("epochs, batch and imgsz must be positive")
    data_file = args.data or build_dataset(args.augment_factor, args.seed)
    check_data(data_file)
    if args.build_only:
        return

    model = load_model(args.model)
    options = dict(
        data=str(data_file), project=str(MODEL / "runs"), name=args.name,
        epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, seed=args.seed,
        patience=25, plots=True, save=True,
        degrees=8, translate=0.1, scale=0.2, fliplr=0.5,
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.3, mosaic=0.3,
        close_mosaic=10, mixup=0.0)
    if args.device:
        options["device"] = args.device
    model.train(**options)
    evaluate(Path(model.trainer.save_dir) / "weights/best.pt", data_file,
             device=args.device)


if __name__ == "__main__":
    main()
