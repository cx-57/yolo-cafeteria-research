"""Recover official food annotations for retained LVIS plate scenes.

Produces additions for experiment.py; never edits images or original plate boxes.
Only explicit edible categories are mapped. Other target types remain unverified.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
from zipfile import ZipFile

from lvis import ARCHIVE_HASHES, ROOT

FOOD_NAMES = {
    "apple", "apricot", "artichoke", "asparagus", "avocado", "banana",
    "bagel", "biscuit_(bread)", "blueberry", "bread", "broccoli", "brownie",
    "burrito", "cabbage", "cake", "carrot", "cauliflower", "celery", "cheese",
    "cherry", "chicken", "chicken_wing", "cookie", "corn", "croissant", "cucumber",
    "cupcake", "doughnut", "egg", "eggplant", "french_fries", "grape", "grapefruit",
    "hamburger", "hotdog", "hot_dog", "ice_cream", "kiwi_fruit", "lemon", "lettuce",
    "lime", "mango", "meat", "muffin", "mushroom", "noodle", "onion", "orange_(fruit)",
    "pancake", "pasta", "peach", "pear", "pepper", "pie", "pineapple", "pizza",
    "potato", "pretzel", "radish", "raspberry", "rice", "salad", "sandwich",
    "sausage", "strawberry", "sushi", "taco", "toast", "tomato", "waffle", "watermelon",
}
COMPOSITE_FOODS = {"pizza", "sandwich", "hamburger", "hotdog", "hot_dog", "salad", "taco", "burrito"}


def iter_array(stream, key, chunk_size=65536):
    """Stream a named array in the official flat LVIS JSON without loading GBs."""
    pattern = re.compile(r'"' + re.escape(key) + r'"\s*:\s*\[')
    buffer = ""
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            raise ValueError(f"Missing JSON array: {key}")
        buffer += chunk
        found = pattern.search(buffer)
        if found:
            buffer = buffer[found.end():]
            break
        buffer = buffer[-256:]
    decoder = json.JSONDecoder()
    while True:
        buffer = buffer.lstrip()
        if buffer.startswith("]"):
            return
        if buffer.startswith(","):
            buffer = buffer[1:].lstrip()
        try:
            value, end = decoder.raw_decode(buffer)
        except json.JSONDecodeError:
            chunk = stream.read(chunk_size)
            if not chunk:
                raise ValueError(f"Truncated JSON array: {key}")
            buffer += chunk
            continue
        yield value
        buffer = buffer[end:]


def archive_array(archive, split, key):
    with ZipFile(archive) as z:
        with z.open(f"lvis_v1_{split}.json") as raw:
            yield from iter_array(io.TextIOWrapper(raw, encoding="utf-8"), key)


def food_box(box, width, height):
    x, y, w, h = map(float, box)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        raise ValueError("Invalid official food box")
    if min(x, y) < -1e-5 or x + w > width + 1e-5 or y + h > height + 1e-5:
        raise ValueError("Official food box lies outside image")
    left, top = max(0, x), max(0, y)
    right, bottom = min(width, x + w), min(height, y + h)
    return (f"0 {(left+right)/2/width:.9f} {(top+bottom)/2/height:.9f} "
            f"{(right-left)/width:.9f} {(bottom-top)/height:.9f}")


def food_portions(annotations, categories):
    """A topping inside a pizza/sandwich is part of that food portion."""
    composites = [a for a in annotations if categories[a["category_id"]] in COMPOSITE_FOODS]
    kept = []
    for ann in annotations:
        if categories[ann["category_id"]] in COMPOSITE_FOODS:
            kept.append(ann)
            continue
        x, y, w, h = map(float, ann["bbox"])
        contained = False
        for parent in composites:
            px, py, pw, ph = map(float, parent["bbox"])
            overlap = max(0, min(x+w, px+pw)-max(x, px)) * max(0, min(y+h, py+ph)-max(y, py))
            if w > 0 and h > 0 and overlap / (w*h) >= .90:
                contained = True
                break
        if not contained:
            kept.append(ann)
    return kept


def recover(folder, manifest, output):
    with manifest.open(newline="") as f:
        records = list(csv.DictReader(f))
    by_split = defaultdict(dict)
    for row in records:
        match = re.fullmatch(r"(train|val)__lvis_(\d+)\.jpg", row["processed_image"])
        if not match:
            raise ValueError("Unexpected retained LVIS filename")
        by_split[match[1]][int(match[2])] = row
    additions, counts = [], Counter()
    for split, selected in sorted(by_split.items()):
        archive = folder / f"lvis_v1_{split}.json.zip"
        with archive.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        if digest != ARCHIVE_HASHES[split]:
            raise ValueError(f"Official archive SHA256 mismatch: {archive}")
        categories = {c["id"]: c["name"] for c in archive_array(archive, split, "categories")}
        food = {cid for cid, name in categories.items() if name in FOOD_NAMES}
        images = {im["id"]: im for im in archive_array(archive, split, "images")
                  if im["id"] in selected}
        if set(images) != set(selected):
            raise ValueError("Some retained plate images have no official metadata")
        rows, provenance, selected_annotations = defaultdict(list), defaultdict(list), defaultdict(list)
        for ann in archive_array(archive, split, "annotations"):
            image_id, cid = ann["image_id"], ann["category_id"]
            if image_id not in selected or cid not in food:
                continue
            selected_annotations[image_id].append(ann)
        for image_id, original_annotations in selected_annotations.items():
            info = images[image_id]
            for ann in food_portions(original_annotations, categories):
                cid = ann["category_id"]
                rows[image_id].append(food_box(ann["bbox"], info["width"], info["height"]))
                provenance[image_id].append({"annotation_id": ann["id"],
                                             "category": categories[cid],
                                             "category_id": cid})
                counts[categories[cid]] += 1
        for image_id, added in sorted(rows.items()):
            additions.append({
                "source": "plate", "image": selected[image_id]["processed_image"],
                "rows": added,
                "provenance": {"source": "Official LVIS v1", "split": split,
                               "archive_sha256": digest, "annotations": provenance[image_id],
                               "food_policy": "Composite portion keeps its box; ingredient >=90% inside a pizza/sandwich/salad portion is not an extra food target.",
                               "all_target_classes_exhaustive": False},
            })
        print(f"{split}: food boxes in {len(rows)} retained plate images", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(additions, indent=2) + "\n")
    print("Food categories and boxes:", dict(counts), flush=True)
    print("Additions:", output, flush=True)
    return additions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, default=ROOT / "data/raw/lvis")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/plate.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    recover(args.archives, args.manifest, args.output)


if __name__ == "__main__":
    main()
