import argparse
import hashlib
import os
import shutil
from pathlib import Path

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Source class name -> final NHS class name.
SOURCE_MAPPINGS = {
    "school_lunch": {
        "Apple": "food",
        "orange": "food",
        "Fish": "food",
        "strawberry": "food",
        "Bread": "food",
        "pineapple": "food",
        "Rice": "food",
        "cucumber": "food",
        "cheese": "food",
        "corn": "food",
        "pasta": "food",
        "butter": "food",
        "Salad": "food",
        "Meat": "food",
        "beans": "food",
        "fried chicken": "food",
        "fried fish": "food",
        "fried meat": "food",
        "muffin": "food",
        "nuggets": "food",
        "potatoes": "food",
        "salad fresh": "food",
        "sausages": "food",
        "Spaghetti": "food",
        "White bread": "food",
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
        "Clamshell": "hinged_container",
    },
    "waste_detection_v3": {
        "Compost-Napkin": "napkin",
        "Compost-Plate-Bowl": "plate",
        "Compost-Utencil": "utensil",
        "Recycle-Plastic-Bottle": "bottle",
        "Trash-Wrapper": "wrapper",
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
    "nhs_local": {
        "food": "food",
        "plate": "plate",
        "paper_boat": "paper_boat",
        "utensil": "utensil",
        "compartment_container": "compartment_container",
        "hinged_container": "hinged_container",
        "napkin": "napkin",
        "drink_carton": "drink_carton",
        "condiment_packet": "condiment_packet",
        "wrapper": "wrapper",
        "bottle": "bottle",
    },
    "nhs_test": {
        "food": "food",
        "plate": "plate",
        "paper_boat": "paper_boat",
        "utensil": "utensil",
        "compartment_container": "compartment_container",
        "hinged_container": "hinged_container",
        "napkin": "napkin",
        "drink_carton": "drink_carton",
        "condiment_packet": "condiment_packet",
        "wrapper": "wrapper",
        "bottle": "bottle",
    },
    "milk_boxes": {
        "ADM": "drink_carton",
        "Alpha": "drink_carton",
        "AlphaIQ": "drink_carton",
        "DauNanh": "drink_carton",
        "GrowPlus": "drink_carton",
        "Hero": "drink_carton",
        "OcCho": "drink_carton",
        "Optimum": "drink_carton",
        "SuSu": "drink_carton",
        "Vinamilk": "drink_carton",
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("source", help="Downloaded YOLO dataset folder")
parser.add_argument("source_name", choices=sorted(SOURCE_MAPPINGS))
parser.add_argument("--output", default="data/processed")
parser.add_argument("--config", default="config.yaml")
args = parser.parse_args()

source = Path(args.source)
output_root = Path(args.output)
output = output_root / args.source_name

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

final_classes = project_config["classes"]
final_ids = {name: int(class_id) for class_id, name in final_classes.items()}

source_mapping = {
    name.strip().lower(): target
    for name, target in SOURCE_MAPPINGS[args.source_name].items()
}

id_mapping = {}
for source_id, source_class in names.items():
    target = source_mapping.get(str(source_class).strip().lower())
    if target in final_ids:
        id_mapping[source_id] = final_ids[target]

if not id_mapping:
    raise SystemExit("None of the approved source classes were found in data.yaml.")

# This folder contains generated hard links and labels, so rebuilding it is safe.
if output.exists():
    shutil.rmtree(output)
(output / "images").mkdir(parents=True)
(output / "labels").mkdir(parents=True)

# Skip exact images already kept from another source.
existing_hashes = set()
for old_image in output_root.glob("*/images/*"):
    if old_image.is_file() and old_image.suffix.lower() in IMAGE_EXTENSIONS:
        existing_hashes.add(hashlib.sha256(old_image.read_bytes()).hexdigest())

local_hashes = {}
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

        if len(parts) != 5:
            bad_rows += 1
            continue

        try:
            source_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            bad_rows += 1
            continue

        if source_id not in id_mapping:
            ignored_classes.add(names.get(source_id, str(source_id)))
            continue

        if not all(0 <= value <= 1 for value in values):
            bad_rows += 1
            continue

        if values[2] <= 0 or values[3] <= 0:
            bad_rows += 1
            continue

        parts[0] = str(id_mapping[source_id])
        new_lines.append(" ".join(parts))

    if not new_lines:
        continue

    image_hash = hashlib.sha256(image.read_bytes()).hexdigest()

    if image_hash in existing_hashes:
        duplicates += 1
        continue

    if image_hash in local_hashes:
        duplicates += 1
        old_label = output / "labels" / f"{local_hashes[image_hash]}.txt"
        old_lines = set(old_label.read_text().splitlines())
        combined = old_lines | set(new_lines)
        kept_instances += len(combined) - len(old_lines)
        old_label.write_text("\n".join(sorted(combined)) + "\n")
        continue

    relative = image.relative_to(source).with_suffix("")
    safe_name = "_".join(relative.parts)
    output_image = output / "images" / f"{safe_name}{image.suffix.lower()}"
    output_label = output / "labels" / f"{safe_name}.txt"

    try:
        os.link(image, output_image)
    except OSError:
        shutil.copy2(image, output_image)

    output_label.write_text("\n".join(new_lines) + "\n")
    local_hashes[image_hash] = safe_name
    kept_images += 1
    kept_instances += len(new_lines)

print(f"Source: {args.source_name}")
print(f"Images scanned: {len(images)}")
print(f"Images kept: {kept_images}")
print(f"Instances kept: {kept_instances}")
print(f"Exact duplicates skipped: {duplicates}")
print(f"Missing labels: {missing_labels}")
print(f"Bad label rows: {bad_rows}")

if ignored_classes:
    print("Ignored classes:", ", ".join(sorted(str(x) for x in ignored_classes)))

print(f"Output: {output}")
