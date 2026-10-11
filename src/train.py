"""Train on the 10 current NHS object classes; never change source labels."""
import argparse
from collections import defaultdict
import hashlib
import os
from pathlib import Path
import re
import shutil

import yaml
from ultralytics import YOLO

from prepare import PROJECT_ROOT, IMAGE_SUFFIXES, LEGACY_SOURCES, current_label

CONFIG = PROJECT_ROOT / "config.yaml"
PROCESSED = PROJECT_ROOT / "data/processed"
OUTPUT = PROJECT_ROOT / "data/training"


def group_for_split(source, filename):
    if source == "condiments":
        return "condiments_same_background"  # Not independent validation images
    if source == "tartanhacks":
        match = re.search(r"IMG_\d+", filename)
        if match:
            return match.group()  # Video frames from same clip stay together
    return filename.split(".rf.")[0]  # Group Roboflow augmented variants


def build_dataset():
    """Generate contiguous labels and an 80/20 split from curated sources only."""
    names = yaml.safe_load(CONFIG.read_text())["classes"]
    names = [names[k] for k in sorted(names)]
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)  # Generated output only; original labels untouched

    groups = defaultdict(list)
    for source in sorted(LEGACY_SOURCES):
        images = PROCESSED / source / "images"
        labels = PROCESSED / source / "labels"
        if not images.is_dir():
            continue
        for image in sorted(images.iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = labels / (image.stem + ".txt")
            if not label.is_file():
                raise FileNotFoundError(f"Missing label for {image}")
            group = source + ":" + group_for_split(source, image.stem)
            groups[group].append((source, image, label))

    if not groups:
        raise ValueError("No curated processed images found")

    counts = {"train": 0, "val": 0}
    for group, items in sorted(groups.items()):
        # Deterministic split; same scene/capture-group never crosses splits.
        fraction = int.from_bytes(hashlib.sha256(group.encode()).digest()[:4], "big") / 2**32
        split = "train" if group.startswith("condiments:") or fraction < 0.8 else "val"
        image_dir = OUTPUT / "images" / split
        label_dir = OUTPUT / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for source, image, label in items:
            image_dest = image_dir / (source + "__" + image.name)
            label_dest = label_dir / (source + "__" + label.stem + ".txt")
            try:
                os.link(image, image_dest)
            except OSError:
                shutil.copy2(image, image_dest)
            content = [current_label(row, source) for row in label.read_text().splitlines() if row.strip()]
            if not content:
                raise ValueError(f"Empty labels in {label}")
            for row in content:
                if int(row.split()[0]) not in range(len(names)):
                    raise ValueError(f"Invalid class ID in {label}")
            label_dest.write_text("\n".join(content) + "\n")
            counts[split] += 1

    if not counts["val"]:
        raise ValueError("No validation images after grouping")
    data = {"path": str(OUTPUT.resolve()), "train": "images/train",
            "val": "images/val", "names": names}
    file = OUTPUT / "data.yaml"
    file.write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"Prepared {counts['train']} training and {counts['val']} validation images.")
    print("These splits are provisional. Final testing requires independent NHS scenes.")
    return file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", nargs="?", type=Path,
                        help="Optional already prepared 10-class data.yaml")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    data_file = args.data or build_dataset()
    expected = yaml.safe_load(CONFIG.read_text())["classes"]
    expected_names = [expected[k] for k in sorted(expected)]
    actual = yaml.safe_load(data_file.read_text())["names"]
    if isinstance(actual, dict):
        actual = [actual[k] if k in actual else actual[str(k)] for k in range(len(expected_names))]
    if actual != expected_names:
        parser.error("Dataset class names/IDs do not match config.yaml (10 classes).")
    if args.build_only:
        print(f"Dataset ready: {data_file}")
        return

    model = YOLO(args.model)
    model.train(data=str(data_file), epochs=args.epochs, imgsz=args.imgsz,
                project="runs", degrees=10, translate=0.10, scale=0.25,
                fliplr=0.5, flipud=0.0, hsv_h=0.01, hsv_s=0.4,
                hsv_v=0.3, mosaic=0.3, mixup=0.0)


if __name__ == "__main__":
    main()
