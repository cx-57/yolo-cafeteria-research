import argparse
import hashlib
import os
import shutil
from pathlib import Path

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Only source classes that are plausible for NHS are allowed through.
SOURCE_MAPPINGS = {
    "school_lunch": {
        "apple": "food",
        "orange": "food",
        "fish": "food",
        "strawberry": "food",
        "bread": "food",
        "pineapple": "food",
        "rice": "food",
        "cucumber": "food",
        "cheese": "food",
        "corn": "food",
        "pasta": "food",
        "butter": "food",
        "salad": "food",
        "meat": "food",
        "beans": "food",
        "fried chicken": "food",
        "fried fish": "food",
        "fried meat": "food",
        "muffin": "food",
        "nuggets": "food",
        "potatoes": "food",
        "salad fresh": "food",
        "sausages": "food",
        "spaghetti": "food",
        "white bread": "food",
        "milk": "drink_carton",
    },
    "food_detection": {
        "pizza": "food",
        "salad": "food",
        "burger": "food",
        "croissant": "food",
        "blueberry-muffin": "food",
        "french-fries": "food",
        "macaroni-cheese": "food",
        "nasi-putih": "food",
        "roti-slice": "food",
        "spaghetti-bolognese": "food",
        "steak": "food",
        "telur-dadar": "food",
    },
    "food_packages": {
        "container with three compartment": "compartment_container",
        "container with two compartments": "compartment_container",
    },
    "macworx": {
        "clamshell": "hinged_container",
    },
    "paper": {
        "paper-plate": "plate",
        "tissue": "napkin",
    },
    "garbage_classification": {
        "bottle": "bottle",
        "plastic utensil": "utensil",
        "milk box": "drink_carton",
        "napkin": "napkin",
        "packet": "condiment_packet",
        "snack package": "wrapper",
    },
    "utensil": {
        "plastic bottle": "bottle",
        "plastic utensil": "utensil",
    },
    "tetra_pak": {
        "tetra pak": "drink_carton",
    },
    "trash_detection": {
        "clear plastic bottle": "bottle",
        "crisp packet": "wrapper",
    },
    "milk_boxes": {
        "adm": "drink_carton",
        "alpha": "drink_carton",
        "alphaiq": "drink_carton",
        "daunanh": "drink_carton",
        "growplus": "drink_carton",
        "hero": "drink_carton",
        "occho": "drink_carton",
        "optimum": "drink_carton",
        "susu": "drink_carton",
        "vinamilk": "drink_carton",
    },
    "nhs_local": {},
    "nhs_test": {},
}

parser = argparse.ArgumentParser()
parser.add_argument("source", help="Downloaded YOLO dataset folder")
parser.add_argument("source_key", choices=sorted(SOURCE_MAPPINGS))
parser.add_argument("--output", default="data/processed")
parser.add_argument("--config", default="config.yaml")
args = parser.parse_args()

source = Path(args.source)
output_root = Path(args.output)
output = output_root / args.source_key

if not source.exists():
    raise SystemExit(f"Source folder not found: {source}")

data_yaml = next(source.rglob("data.yaml"), None)
if data_yaml is None:
    raise SystemExit("No data.yaml found in source folder.")

with open(data_yaml) as f:
    source_config = yaml.safe_load(f)

with open(args.config) as f:
    project_config = yaml.safe_load(f)

names = source_config.get("names", {})
if isinstance(names, list):
    names = {i: name for i, name in enumerate(names)}
else:
    names = {int(i): name for i, name in names.items()}

final_ids = {
    name: int(class_id)
    for class_id, name in project_config["classes"].items()
}

if args.source_key in {"nhs_local", "nhs_test"}:
    approved = {name: name for name in final_ids}
else:
    approved = SOURCE_MAPPINGS[args.source_key]

id_mapping = {}
for source_id, source_class in names.items():
    target = approved.get(str(source_class).strip().lower())
    if target in final_ids:
        id_mapping[source_id] = final_ids[target]

if not id_mapping:
    raise SystemExit("None of the approved source classes were found in data.yaml.")

# Rebuilding a source starts from its mechanical candidates again.
if output.exists():
    shutil.rmtree(output)
(output / "images").mkdir(parents=True)
(output / "labels").mkdir(parents=True)

# Exact duplicates from earlier sources point to the label file already kept.
existing = {}
for old_image in output_root.glob("*/images/*"):
    if not old_image.is_file() or old_image.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    image_hash = hashlib.sha256(old_image.read_bytes()).hexdigest()
    old_label = old_image.parent.parent / "labels" / f"{old_image.stem}.txt"
    existing[image_hash] = old_label

local = {}
kept_images = 0
kept_instances = 0
duplicates = 0
bad_rows = 0
missing_labels = 0
ignored_classes = set()

images = sorted(
    path for path in source.rglob("*")
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
)

for image in images:
    parts = list(image.parts)

    if "images" in parts:
        i = parts.index("images")
        label_parts = parts[:]
        label_parts[i] = "labels"
        label = Path(*label_parts).with_suffix(".txt")
    else:
        label = image.with_suffix(".txt")

    if not label.exists():
        missing_labels += 1
        continue

    new_lines = []

    for line in label.read_text().splitlines():
        parts = line.split()

        if len(parts) < 5:
            bad_rows += 1
            continue

        try:
            source_id = int(float(parts[0]))
            raw_values = [float(value) for value in parts[1:]]
        except ValueError:
            bad_rows += 1
            continue

        if source_id not in id_mapping:
            ignored_classes.add(names.get(source_id, str(source_id)))
            continue

        # Roboflow YOLO exports can contain either detection boxes
        # (class x_center y_center width height) or segmentation polygons
        # (class x1 y1 x2 y2 ...). Convert polygons to axis-aligned boxes
        # because this project trains an object detector.
        if len(raw_values) == 4:
            x, y, width, height = raw_values
        elif len(raw_values) >= 6 and len(raw_values) % 2 == 0:
            xs = raw_values[0::2]
            ys = raw_values[1::2]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            x = (x_min + x_max) / 2
            y = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min
        else:
            bad_rows += 1
            continue

        values = [x, y, width, height]

        if not all(0 <= value <= 1 for value in values):
            bad_rows += 1
            continue

        if width <= 0 or height <= 0:
            bad_rows += 1
            continue

        new_lines.append(
            f"{id_mapping[source_id]} {x:.8f} {y:.8f} {width:.8f} {height:.8f}"
        )

    if not new_lines:
        continue

    image_hash = hashlib.sha256(image.read_bytes()).hexdigest()

    if image_hash in existing:
        duplicates += 1
        old_label = existing[image_hash]

        if old_label.exists():
            old_lines = set(old_label.read_text().splitlines())
            combined = old_lines | set(new_lines)
            kept_instances += len(combined) - len(old_lines)
            old_label.write_text("\n".join(sorted(combined)) + "\n")

        continue

    if image_hash in local:
        duplicates += 1
        old_label = output / "labels" / f"{local[image_hash]}.txt"
        old_lines = set(old_label.read_text().splitlines())
        combined = old_lines | set(new_lines)
        kept_instances += len(combined) - len(old_lines)
        old_label.write_text("\n".join(sorted(combined)) + "\n")
        continue

    relative = image.relative_to(source).with_suffix("")
    safe_name = "__".join(relative.parts)
    output_image = output / "images" / f"{safe_name}{image.suffix.lower()}"
    output_label = output / "labels" / f"{safe_name}.txt"

    try:
        os.link(image, output_image)
    except OSError:
        shutil.copy2(image, output_image)

    output_label.write_text("\n".join(new_lines) + "\n")
    local[image_hash] = safe_name
    kept_images += 1
    kept_instances += len(new_lines)

print(f"Source: {args.source_key}")
print(f"Images scanned: {len(images)}")
print(f"Images kept: {kept_images}")
print(f"Instances kept: {kept_instances}")
print(f"Exact duplicates skipped: {duplicates}")
print(f"Missing labels: {missing_labels}")
print(f"Bad label rows: {bad_rows}")

if ignored_classes:
    print("Ignored classes:", ", ".join(sorted(str(x) for x in ignored_classes)))

print(f"Output: {output}")
