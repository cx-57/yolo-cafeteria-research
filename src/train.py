"""Build leakage-aware training views, train YOLO11, and evaluate on untouched validation photos.

10x means 10 training *views* of each original. The views hardlink the same
image bytes; Ultralytics independently augments each view while training.
No synthetic images are committed, and validation is never augmented.
"""
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

from prepare import (IMAGE_SUFFIXES, LEGACY_SOURCES, PROJECT_ROOT,
                     current_label, remap_label_line)

CONFIG = PROJECT_ROOT / "config.yaml"
PROCESSED = PROJECT_ROOT / "data/processed"
OUTPUT = PROJECT_ROOT / "data/training"
# These new uploads already use current IDs. Do not remap them as legacy.
TRAIN_ONLY = {
    "condiments", "tartanhacks", "waste_detection_v3", "dry_wet_waste",
    "school_lunch_asian", "food_american",
}
SOURCES = LEGACY_SOURCES | TRAIN_ONLY


def class_names():
    classes = {int(k): v for k, v in yaml.safe_load(CONFIG.read_text())["classes"].items()}
    if sorted(classes) != list(range(len(classes))):
        raise ValueError("config.yaml class IDs must be contiguous, starting at zero")
    return [classes[i] for i in range(len(classes))]


def group_for_split(source, filename):
    """Keep Roboflow variants of one original together, regardless of split prefix."""
    stem = re.sub(r"^(train|valid|val|test)__", "", filename)
    stem = stem.split(".rf.")[0]
    if source == "tartanhacks":
        match = re.search(r"IMG_\d+", stem)
        if match:
            return match.group()
    return stem


def pixel_digest(image):
    """Match byte-different encodings of the same exact pixels across sources."""
    with Image.open(image) as opened:
        rgb = opened.convert("RGB")
        return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def read_labels(label, source, names):
    id_to_name = dict(enumerate(names))
    name_to_id = {name: idx for idx, name in enumerate(names)}
    rows = []
    if not label.is_file():
        raise FileNotFoundError("Missing YOLO annotation: " + str(label))
    for number, line in enumerate(label.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label}:{number}: expected 5 YOLO detection values")
        converted = current_label(line, source)
        valid = remap_label_line(converted, id_to_name, name_to_id, label, number)
        if valid is None:
            raise ValueError(f"{label}:{number}: unmapped class")
        rows.append(valid)
    if not rows:
        raise ValueError("Empty annotations (unverified negatives are not allowed): " + str(label))
    return rows


def collect(names):
    """Validate every curated original and keep exact duplicate images in one split."""
    grouped = defaultdict(list)
    digest_groups = defaultdict(set)
    per_source = Counter()
    for source in sorted(SOURCES):
        image_dir = PROCESSED / source / "images"
        label_dir = PROCESSED / source / "labels"
        if not image_dir.is_dir():
            continue
        for image in sorted(image_dir.iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = label_dir / (image.stem + ".txt")
            rows = read_labels(label, source, names)
            digest = pixel_digest(image)  # Also checks that the image decodes.
            group = source + ":" + group_for_split(source, image.stem)
            grouped[group].append((source, image, rows))
            digest_groups[digest].add(group)
            per_source[source] += 1
        # A label with no image should not silently pass.
        for label in label_dir.glob("*.txt"):
            if not any((image_dir / (label.stem + ext)).is_file()
                       for ext in IMAGE_SUFFIXES):
                raise ValueError("Orphan annotation: " + str(label))
    if not grouped:
        raise ValueError("No images found in curated processed sources")

    parent = {group: group for group in grouped}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for keys in digest_groups.values():
        keys = sorted(keys)
        for key in keys[1:]:
            union(keys[0], key)

    components = defaultdict(list)
    for key, items in grouped.items():
        components[find(key)].extend(items)
    return components, dict(per_source), sum(len(keys) > 1 for keys in digest_groups.values())


def build_dataset(augment_factor=10, seed=42):
    """Make grouped 80/20 source views; augmented copies are TRAIN-ONLY."""
    if augment_factor < 1 or augment_factor > 20:
        raise ValueError("--augment-factor must be between 1 and 20")
    names = class_names()
    groups, source_counts, duplicate_groups = collect(names)
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)  # Generated and Git-ignored, never the source corpus.

    counts = Counter()
    class_images = {"train": Counter(), "val": Counter()}
    class_boxes = {"train": Counter(), "val": Counter()}
    groups_by_split = Counter()
    for key, items in sorted(groups.items()):
        fraction = int.from_bytes(hashlib.sha256((str(seed) + ":" + key).encode()).digest()[:8], "big") / 2**64
        must_train = any(source in TRAIN_ONLY for source, _, _ in items)
        split = "train" if must_train or fraction < 0.8 else "val"
        groups_by_split[split] += 1
        image_dir = OUTPUT / "images" / split
        label_dir = OUTPUT / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for source, image, rows in items:
            stem = source + "__" + image.stem
            output_image = image_dir / (stem + image.suffix)
            output_label = label_dir / (stem + ".txt")
            if output_image.exists() or output_label.exists():
                raise ValueError("Destination name collision: " + stem)
            link_or_copy(image, output_image)
            output_label.write_text("\n".join(rows) + "\n")
            counts[split + "_originals"] += 1
            for idx in set(int(row.split()[0]) for row in rows):
                class_images[split][idx] += 1
            for row in rows:
                class_boxes[split][int(row.split()[0])] += 1

            # Distinct paths, same bytes/inode. YOLO's random transformations
            # independently augment every view during each training epoch.
            if split == "train":
                for view in range(1, augment_factor):
                    extra_stem = stem + f"__view{view:02d}"
                    extra_image = image_dir / (extra_stem + image.suffix)
                    extra_label = label_dir / (extra_stem + ".txt")
                    # Hardlink only, so 10x does not multiply disk consumption.
                    os.link(output_image, extra_image)
                    os.link(output_label, extra_label)
                counts["train_views"] += augment_factor
    if not counts["train_originals"] or not counts["val_originals"]:
        raise ValueError("Train and validation must each contain original images")

    data = {"path": str(OUTPUT.resolve()), "train": "images/train",
            "val": "images/val", "names": names}
    yaml_path = OUTPUT / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False))
    report = {
        "source_images": source_counts, "original_images": sum(source_counts.values()),
        "augment_factor": augment_factor, "seed": seed,
        "train_originals": counts["train_originals"],
        "train_views": counts["train_views"],
        "val_originals": counts["val_originals"],
        "exact_pixel_duplicate_groups_linked": duplicate_groups,
        "group_counts": dict(groups_by_split),
        "train_class_images_original": {name: class_images["train"][i] for i, name in enumerate(names)},
        "val_class_images_original": {name: class_images["val"][i] for i, name in enumerate(names)},
        "train_class_boxes_original": {name: class_boxes["train"][i] for i, name in enumerate(names)},
        "val_class_boxes_original": {name: class_boxes["val"][i] for i, name in enumerate(names)},
        "missing_train_classes": [name for i, name in enumerate(names) if not class_boxes["train"][i]],
        "missing_val_classes": [name for i, name in enumerate(names) if not class_boxes["val"][i]],
        "caveat": "Public-source validation is provisional. Same-scene near-duplicates, "
                  "incomplete multiclass annotations and domain shift still require review. "
                  "10x views do not create independent examples. Never report final accuracy "
                  "without separate, held-out NHS cafeteria scenes.",
    }
    (OUTPUT / "split_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Originals: {counts['train_originals']} train / {counts['val_originals']} val; "
          f"{counts['train_views']} training views ({augment_factor}x); val never augmented.")
    print("Missing in training:", report["missing_train_classes"])
    print("Missing in validation:", report["missing_val_classes"])
    print("Source-independent NHS test scenes are still REQUIRED.")
    return yaml_path


def check_names(data_file):
    data = yaml.safe_load(Path(data_file).read_text())
    actual = data["names"]
    if isinstance(actual, dict):
        actual = [actual.get(i, actual.get(str(i))) for i in range(len(class_names()))]
    if actual != class_names():
        raise ValueError("Dataset names/IDs must exactly match config.yaml (11 classes).")


def evaluate(weights, data_file, imgsz=640, device=None):
    """Evaluate only original validation images; save interpretable per-class results."""
    from ultralytics import YOLO

    check_names(data_file)
    model = YOLO(str(weights))
    run_folder = PROJECT_ROOT / "runs"
    options = {"data": str(data_file), "split": "val", "imgsz": imgsz,
               "project": str(run_folder), "name": "evaluation",
               "plots": True, "exist_ok": False}
    if device:
        options["device"] = device
    metrics = model.val(**options)
    names = class_names()
    by_id = {
        int(idx): {"ap50": float(a50), "map50_95": float(a)}
        for idx, a50, a in zip(metrics.box.ap_class_index,
                             metrics.box.ap50, metrics.box.ap)
    }
    result = {"precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
              "map50": float(metrics.box.map50),
              "map50_95": float(metrics.box.map),
              "per_class": {name: by_id.get(i) for i, name in enumerate(names)},
              "note": "null means class not measurable in this provisional validation split; "
                      "these metrics are NOT independent NHS test performance."}
    save_folder = Path(metrics.save_dir)
    (save_folder / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Validation results:", save_folder / "metrics.json")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", type=Path,
                        help="Optional reviewed 11-class data.yaml; bypasses corpus build")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--augment-factor", type=int, default=10,
                        help="Training views per original (default 10, validation 1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", help="e.g. mps for Apple Silicon, 0 for CUDA, cpu")
    parser.add_argument("--name", default="yolo11n-10x")
    parser.add_argument("--evaluate", type=Path, metavar="BEST_PT",
                        help="Evaluate an existing checkpoint using prepared data.yaml")
    parser.add_argument("--resume", type=Path, metavar="LAST_PT",
                        help="Resume a stopped Ultralytics training run")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 32:
        parser.error("epochs, batch and imgsz must be positive")
    if args.evaluate and args.resume:
        parser.error("--evaluate and --resume cannot be combined")

    if args.resume:
        if not (OUTPUT / "data.yaml").exists():
            parser.error("Missing prepared data; run --build-only with the SAME settings first")
        from ultralytics import YOLO
        YOLO(str(args.resume)).train(resume=True)
        return

    if args.evaluate:
        data_file = args.data or OUTPUT / "data.yaml"
        evaluate(args.evaluate, data_file, args.imgsz, args.device)
        return

    data_file = args.data or build_dataset(args.augment_factor, args.seed)
    check_names(data_file)
    if args.build_only:
        print("Prepared:", data_file)
        return

    from ultralytics import YOLO
    model = YOLO(args.model)
    settings = dict(data=str(data_file), epochs=args.epochs, imgsz=args.imgsz,
                    batch=args.batch, workers=args.workers, seed=args.seed,
                    patience=25, project=str(PROJECT_ROOT / "runs"), name=args.name,
                    exist_ok=False, plots=True, save=True,
                    degrees=8, translate=0.10, scale=0.20,
                    fliplr=0.5, flipud=0, hsv_h=0.015, hsv_s=0.4,
                    hsv_v=0.3, mosaic=0.3, close_mosaic=10, mixup=0.0)
    if args.device:
        settings["device"] = args.device
    model.train(**settings)
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError("Training finished without best.pt: " + str(best))
    evaluate(best, data_file, args.imgsz, args.device)


if __name__ == "__main__":
    main()
