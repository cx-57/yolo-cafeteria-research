"""Stage COCO-annotated images for visual screening, not independent evaluation."""
import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

HINTS = {
    "bottle": {"bottle", "plastic bottle", "pet bottle", "clear plastic bottle"},
    "utensil": {"spoon", "plastic spoon", "fork", "plastic fork", "plastic utensils", "cutlery"},
    "wrapper": {"packet", "crisp packet", "food wrapper", "plastic wrapper", "other plastic wrapper", "snack wrapper"},
    "drink_carton": {"drink carton", "milk carton", "juice carton", "beverage carton", "tetra pak", "tetrapak"},
    "hinged_container": {"hinged container", "clamshell container", "clamshell food container"},
}

def norm(s):
    return " ".join(str(s).lower().replace("_", " ").replace("-", " ").split())

def collect(annotation_file, image_root):
    coco = json.loads(annotation_file.read_text(encoding="utf-8"))
    categories = {c["id"]: norm(c["name"]) for c in coco["categories"]}
    images = {im["id"]: im for im in coco["images"]}
    hints = defaultdict(set)
    for row in coco["annotations"]:
        category = categories.get(row.get("category_id"), "")
        for target, aliases in HINTS.items():
            if category in aliases and row.get("image_id") in images:
                hints[row["image_id"]].add(target)
    items = []
    for image_id in sorted(hints, key=str):
        filename = Path(images[image_id]["file_name"])
        if filename.is_absolute() or ".." in filename.parts:
            continue
        path = image_root / filename
        if not path.is_file():
            path = image_root / filename.name
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            items.append((path, sorted(hints[image_id]), str(filename)))
    return items, sorted(set(categories.values()))

def make_sheet(rows, path):
    try:
        from PIL import Image, ImageDraw, ImageOps
    except ImportError:
        return False
    columns, w, h = 6, 190, 185
    canvas = Image.new("RGB", (columns*w, (1 + (len(rows)+5)//6)*h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), "CANDIDATES ONLY: inspect individually. NOT independent test data.", fill="black")
    for i, row in enumerate(rows):
        x, y = (i%columns)*w, (1+i//columns)*h
        try:
            with Image.open(row["path"]) as f:
                im = ImageOps.exif_transpose(f).convert("RGB")
                im.thumbnail((w-10, h-42))
            canvas.paste(im, (x+(w-im.width)//2, y+3))
            draw.text((x+5, y+h-36), row["filename"][:27], fill="black")
            draw.text((x+5, y+h-19), ", ".join(row["hints"])[:30], fill="black")
        except (OSError, ValueError):
            draw.text((x+5, y+20), "UNREADABLE", fill="red")
    canvas.save(path, quality=88)
    return True

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", required=True, type=Path)
    parser.add_argument("--images-root", required=True, type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--per-class", type=int, default=14)
    parser.add_argument("--total", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Refusing to overwrite an existing output folder")
    if args.total < 1 or args.per_class < 1:
        parser.error("Counts must be positive")
    candidates, source_categories = collect(args.coco, args.images_root)
    random.Random(args.seed).shuffle(candidates)
    selected = []
    counts = {key: 0 for key in HINTS}
    for path, hints, original in candidates:
        if len(selected) >= args.total:
            break
        if not any(counts[h] < args.per_class for h in hints):
            continue
        selected.append((path, hints, original))
        for h in hints:
            counts[h] += 1
    if not selected:
        parser.error("No matching files. COCO labels: " + str(source_categories))
    args.out.mkdir(parents=True)
    image_folder = args.out / "images"
    image_folder.mkdir()
    seen, records = set(), []
    for path, hints, original in selected:
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if sha in seen:
            continue
        seen.add(sha)
        filename = f"{len(records)+1:03d}_{sha[:12]}{path.suffix.lower()}"
        shutil.copyfile(path, image_folder / filename)
        records.append(dict(filename=filename, path=image_folder/filename, hints=hints,
                            source_original=original, sha256=sha))
    fields = ["filename", "source_original", "sha256", "source_hints",
              "source", "source_url", "license", "visual_status", "reason",
              "historical_exposure", "source_group", "independent", "ground_truth"]
    with (args.out / "review.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow(dict(filename=r["filename"], source_original=r["source_original"],
                sha256=r["sha256"], source_hints=";".join(r["hints"]),
                source=args.source, source_url=args.source_url, license=args.license,
                visual_status="PENDING", reason="", historical_exposure="UNVERIFIED",
                source_group="UNVERIFIED", independent="NO", ground_truth="NOT_LABELED"))
    make_sheet(records, args.out / "contact-sheet.jpg")
    (args.out / "README.txt").write_text(
        "PROVISIONAL REVIEW ONLY. COCO labels are source hints, not accepted truth.\n"
        "Inspect each image; check exact/near duplicates and capture history on the Mac.\n"
        "Do not claim independence or run inference until provenance, membership, and annotation rules are frozen.\n"
        "No human disposal labels have been generated.\n", encoding="utf-8")
    counts = {key: sum(key in r["hints"] for r in records) for key in HINTS}
    print(json.dumps(dict(candidates=len(records), class_hints=counts,
        source_categories=source_categories, independent_verified=0,
        bin_labels=0, output=str(args.out)), indent=2))

if __name__ == "__main__":
    main()
