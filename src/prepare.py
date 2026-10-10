import argparse
import math
import os
import shutil
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "data" / "raw"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
CANDIDATE_ROOT = PROJECT_ROOT / "data" / "candidates"

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
        "milk": 7,
    },
    "tetra_pak": {"tetra pak": 7},
    "container": {"clamshell": 5},
    "trash": {
        "clear plastic bottle": 10,
        "crisp packet": 9,
    },
    "utensils": {"plastic utensil": 3},
    "condiments": {
        "chili": 8,
        "tomato": 8,
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


def class_names_by_id(names):
    if isinstance(names, dict):
        return {int(class_id): name for class_id, name in names.items()}
    if isinstance(names, list):
        return dict(enumerate(names))
    raise ValueError("data.yaml 'names' must be a list or a mapping")


def remap_label_line(line, names, dataset_map, label_path, line_number):
    parts = line.split()
    if not parts:
        return None

    where = f'{label_path}:{line_number}'
    try:
        source_id = int(parts[0])
        source_class = str(names[source_id]).strip().lower()
        coordinates = [float(value) for value in parts[1:]]
    except (ValueError, KeyError) as exc:
        raise ValueError(f'{where}: invalid or unknown source class/coordinates') from exc
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in coordinates):
        raise ValueError(f'{where}: coordinates must be finite and within [0, 1]')
    if source_class not in dataset_map:
        return None

    coordinate_text = parts[1:]
    if len(coordinates) == 4:
        box = coordinate_text
    elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:
        x_coordinates = coordinates[0::2]
        y_coordinates = coordinates[1::2]
        x_min, x_max = min(x_coordinates), max(x_coordinates)
        y_min, y_max = min(y_coordinates), max(y_coordinates)
        box = [
            format((x_min + x_max) / 2, ".8g"),
            format((y_min + y_max) / 2, ".8g"),
            format(x_max - x_min, ".8g"),
            format(y_max - y_min, ".8g"),
        ]
    else:
        raise ValueError(
            f"{label_path}:{line_number}: expected a YOLO box or polygon"
        )

    new_id = dataset_map[source_class]
    x, y, width, height = map(float, box)
    if (width <= 0 or height <= 0 or x-width/2 < -1e-6
            or y-height/2 < -1e-6 or x+width/2 > 1+1e-6
            or y+height/2 > 1+1e-6):
        raise ValueError(f'{where}: box must have positive size and lie inside the image')
    return " ".join([str(new_id), *box])


def process_dataset(dataset, processed_root=PROCESSED_ROOT, source_key=None):
    dataset = Path(dataset)
    processed_root = Path(processed_root)
    source_key = source_key or dataset.name
    if source_key not in CLASS_MAPS:
        raise ValueError(f'No class map configured for source: {source_key}')
    dataset_map = CLASS_MAPS[source_key]
    with (dataset / "data.yaml").open() as file:
        data = yaml.safe_load(file)
    names = class_names_by_id(data["names"])

    processed_name = PROCESSED_NAMES.get(source_key, source_key)
    kept_images = 0

    for image in sorted(dataset.rglob("*")):
        if (
            image.suffix.lower() not in IMAGE_SUFFIXES
            or image.parent.name != "images"
        ):
            continue

        label = image.parent.parent / "labels" / f"{image.stem}.txt"
        if not label.exists():
            continue

        cleaned = []
        for line_number, line in enumerate(label.read_text().splitlines(), start=1):
            remapped = remap_label_line(
                line, names, dataset_map, label, line_number
            )
            if remapped is not None:
                cleaned.append(remapped)

        if not cleaned:
            continue

        split = image.parent.parent.name
        image_folder = processed_root / processed_name / "images"
        label_folder = processed_root / processed_name / "labels"
        image_folder.mkdir(parents=True, exist_ok=True)
        label_folder.mkdir(parents=True, exist_ok=True)

        output_image = image_folder / f'{split}__{image.name}'
        if not output_image.exists():
            try:
                os.link(image, output_image)
            except OSError:
                shutil.copy2(image, output_image)
        elif output_image.read_bytes() != image.read_bytes():
            raise ValueError(f'Output image collision: {output_image}')

        output_label = label_folder / f'{split}__{image.stem}.txt'
        text = '\n'.join(cleaned) + '\n'
        if output_label.exists() and output_label.read_text() != text:
            raise ValueError(f'Refusing to overwrite different labels: {output_label}')
        output_label.write_text(text)
        kept_images += 1

    return kept_images


def main():
    parser = argparse.ArgumentParser(description='Remap raw YOLO exports into candidates for visual review.')
    parser.add_argument('dataset', nargs='?', type=Path, help='Raw dataset containing data.yaml')
    parser.add_argument('source_key', nargs='?', choices=sorted(CLASS_MAPS), metavar='source_key')
    parser.add_argument('--output-root', type=Path, default=CANDIDATE_ROOT)
    args = parser.parse_args()
    try:
        if args.dataset is not None:
            key = args.source_key or args.dataset.name
            count = process_dataset(args.dataset, args.output_root, source_key=key)
            print(f'{key}: kept {count} candidate images; visual review required')
        else:
            if args.source_key:
                parser.error('source_key requires a dataset path')
            if not SOURCE_ROOT.is_dir():
                parser.error(f'No raw datasets found at {SOURCE_ROOT}; provide a dataset path')
            for dataset in sorted(SOURCE_ROOT.iterdir()):
                if dataset.is_dir():
                    count = process_dataset(dataset, args.output_root)
                    print(f'{dataset.name}: kept {count} candidate images; visual review required')
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

