"""Convert downloaded Roboflow labels into review candidates.

Original sources and accepted dataset images are never overwritten.
"""
import argparse
import math
import os
from pathlib import Path
import shutil

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "data/raw"
PROCESSED_ROOT = PROJECT_ROOT / "data/processed"
CANDIDATE_ROOT = PROJECT_ROOT / "data/candidates"

CLASS_MAPS = {
    "school_lunch": {
        "apple": 0,
        "orange": 0,
        "fish": 0,
        "strawberry": 0,
        "bread": 0,
        "pineapple": 0,
        "rice": 0,
        "cucumber": 0,
        "cheese": 0,
        "corn": 0,
        "pasta": 0,
        "butter": 0,
        "salad": 0,
        "meat": 0,
        "beans": 0,
        "fried chicken": 0,
        "fried fish": 0,
        "fried meat": 0,
        "muffin": 0,
        "nuggets": 0,
        "potatoes": 0,
        "salad fresh": 0,
        "sausages": 0,
        "spaghetti": 0,
        "white bread": 0,
        "milk": 6,
    },
    "tetra_pak": {"tetra pak": 6},
    "container": {"clamshell": 4},
    "trash": {
        "clear plastic bottle": 9,
        "crisp packet": 8,
    },
    "utensils": {"plastic utensil": 3},
    "condiments": {
        "chili": 7,
        "tomato": 7,
    },
}

PROCESSED_NAMES = {
    "utensils": "utensil",
}
# Explicit food categories present in the recovered School Lunch v2 export.
# Packaged cup desserts, drinkable yogurt, and the ambiguous 'pas' remain
# unmapped: their packaging/object class needs image-level review.
CLASS_MAPS['school_lunch'].update(dict.fromkeys([
    'cherry tomatoes', 'citrus', 'curry', 'fried noodles', 'mixed rice',
    'soups', 'spicy chili-flavored tofu', 'udon', 'avocado', 'boiled eggs',
    'borsch', 'cakes', 'casserole with meat and vegetables',
    'cereal based cooked food', 'cooked eggplant', 'cooked food based on meat',
    'cooked food meat with vegetables', 'cookies', 'crepe', 'crepe w filling',
    'cutlet', 'desserts', 'egg product', 'fried cheese', 'fried eggs',
    'grapefruit', 'green olives', 'kiwi', 'lavash', 'lemon', 'lime', 'mandarin',
    'mango', 'manty', 'meat based soup', 'meat product', 'melon',
    'minced meat shashlyk', 'noodles soup', 'onion rings', 'pancakes',
    'pastry', 'peach', 'pear', 'pickled cucumber', 'pie', 'plov', 'porridge',
    'salad with sauce', 'seafood', 'snacks', 'soup-plain', 'tushpara-fried',
    'tushpara-wo-soup', 'vareniki', 'vegetable based cooked food',
    'vegetable soup', 'watermelon', 'wings',
], 0))
CLASS_MAPS['macworx'] = CLASS_MAPS['container']
CLASS_MAPS['utensil'] = CLASS_MAPS['utensils']
CLASS_MAPS['trash_detection'] = CLASS_MAPS['trash']
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

# Previously committed processed labels use the original 11-class IDs.
# Keep those source labels unchanged for research provenance; audit/train
# convert them to the new 10-class IDs when reading them.
LEGACY_TO_CURRENT = {0: 0, 1: 1, 2: 2, 3: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9}
LEGACY_SOURCES = {
    "condiments", "plate", "macworx", "school_lunch",
    "tartanhacks", "tetra_pak", "trash_detection", "utensil",
}


def current_label(line, source):
    """Map the stored historical class IDs to the active 11-class scheme."""
    parts = line.split()
    if source in LEGACY_SOURCES:
        old_id = int(parts[0])
        if old_id not in LEGACY_TO_CURRENT:
            raise ValueError(f"Unsupported old class ID {old_id} in {source}")
        parts[0] = str(LEGACY_TO_CURRENT[old_id])
    return " ".join(parts)


def class_names_by_id(names):
    return {int(i): name for i, name in
            (enumerate(names) if isinstance(names, list) else names.items())}


def remap_label_line(line, names, dataset_map, label_path, line_number):
    """Read YOLO box/polygon, check geometry, and map a source class."""
    parts = line.split()
    location = f"{label_path}:{line_number}"
    try:
        old_id = int(parts[0])
        name = str(names[old_id]).strip().lower()
        coords = [float(value) for value in parts[1:]]
    except (ValueError, KeyError, IndexError) as exc:
        raise ValueError(f"{location}: invalid class or coordinates") from exc
    if (len(coords) != 4 and (len(coords) < 6 or len(coords) % 2)) or not all(
        math.isfinite(x) and 0 <= x <= 1 for x in coords
    ):
        raise ValueError(f"{location}: invalid YOLO coordinates")
    if name not in dataset_map:
        return None
    if len(coords) == 4:
        x, y, w, h = coords
        values = parts[1:]  # Preserve source precision.
    else:
        left, right = min(coords[::2]), max(coords[::2])
        top, bottom = min(coords[1::2]), max(coords[1::2])
        x, y, w, h = (left+right)/2, (top+bottom)/2, right-left, bottom-top
        values = [f"{value:.8g}" for value in (x, y, w, h)]
    if (w <= 0 or h <= 0 or x-w/2 < -1e-6 or y-h/2 < -1e-6
            or x+w/2 > 1+1e-6 or y+h/2 > 1+1e-6):
        raise ValueError(f"{location}: bounding box outside image")
    return " ".join([str(dataset_map[name]), *values])


def process_dataset(dataset, processed_root=CANDIDATE_ROOT, source_key=None):
    """Save only relevant source images and corresponding review labels."""
    dataset, processed_root = Path(dataset), Path(processed_root)
    source_key = source_key or dataset.name
    if source_key not in CLASS_MAPS:
        raise ValueError(f"No class map for {source_key}")
    names = class_names_by_id(yaml.safe_load((dataset / "data.yaml").read_text())["names"])
    dest = processed_root / PROCESSED_NAMES.get(source_key, source_key)
    count = 0
    for image in sorted(dataset.rglob("*")):
        if image.suffix.lower() not in IMAGE_SUFFIXES or image.parent.name != "images":
            continue
        original_label = image.parent.parent / "labels" / (image.stem + ".txt")
        if not original_label.exists():
            continue
        rows = [remap_label_line(line, names, CLASS_MAPS[source_key], original_label, n)
                for n, line in enumerate(original_label.read_text().splitlines(), 1) if line.strip()]
        rows = [row for row in rows if row]
        if not rows:
            continue
        split = image.parent.parent.name
        base = split + "__" + image.stem
        output_image = dest / "images" / (split + "__" + image.name)
        output_label = dest / "labels" / (base + ".txt")
        output_image.parent.mkdir(parents=True, exist_ok=True)
        output_label.parent.mkdir(parents=True, exist_ok=True)
        if output_image.exists() and output_image.read_bytes() != image.read_bytes():
            raise ValueError(f"Conflicting image: {output_image}")
        if not output_image.exists():
            try:
                os.link(image, output_image)
            except OSError:
                shutil.copy2(image, output_image)
        label_text = "\n".join(rows) + "\n"
        if output_label.exists() and output_label.read_text() != label_text:
            raise ValueError(f"Conflicting label: {output_label}")
        output_label.write_text(label_text)
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Downloaded Roboflow dataset directory")
    parser.add_argument("source_key", choices=sorted(CLASS_MAPS))
    args = parser.parse_args()
    count = process_dataset(args.dataset, source_key=args.source_key)
    print(f"{count} images added to data/candidates for visual review")


if __name__ == "__main__":
    main()
