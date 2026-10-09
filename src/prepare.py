import os
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "data" / "raw"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"

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

    source_id = int(parts[0])
    source_class = str(names[source_id]).strip().lower()
    if source_class not in dataset_map:
        return None

    coordinate_text = parts[1:]
    coordinates = [float(value) for value in coordinate_text]
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
    return " ".join([str(new_id), *box])


def process_dataset(dataset, processed_root=PROCESSED_ROOT):
    dataset_map = CLASS_MAPS[dataset.name]
    with (dataset / "data.yaml").open() as file:
        data = yaml.safe_load(file)
    names = class_names_by_id(data["names"])

    processed_name = PROCESSED_NAMES.get(dataset.name, dataset.name)
    kept_images = 0

    for image in dataset.rglob("*"):
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
        image_folder = processed_root / processed_name / split / "images"
        label_folder = processed_root / processed_name / split / "labels"
        image_folder.mkdir(parents=True, exist_ok=True)
        label_folder.mkdir(parents=True, exist_ok=True)

        output_image = image_folder / image.name
        if not output_image.exists():
            os.link(image, output_image)

        (label_folder / f"{image.stem}.txt").write_text(
            "\n".join(cleaned) + "\n"
        )
        kept_images += 1

    return kept_images


def main():
    for dataset in sorted(SOURCE_ROOT.iterdir()):
        if not dataset.is_dir():
            continue
        if dataset.name not in CLASS_MAPS:
            raise ValueError(
                f"No class map configured for raw dataset: {dataset.name}"
            )
        kept_images = process_dataset(dataset)
        print(f"{dataset.name}: kept {kept_images} images")
    print("Done!")


if __name__ == "__main__":
    main()
