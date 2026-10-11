"""Download LVIS paper-plate photos and YOLO labels into review candidates.

Run: python3 src/lvis.py
Nothing here is added to the accepted training corpus automatically.
"""
import argparse
import csv
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import shutil
from urllib.request import urlretrieve
from zipfile import ZipFile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PLATE_ID = 756                 # LVIS v1 paper_plate
YOLO_ID = 1                   # this project's plate class
ARCHIVES = {
    "train": "334a4caa374030a7817cf050364525e910f7960f9b6968cef47cffbf3893f8ba",
    "val": "5cae9a3c79aadb667550c2b5dcf7f4d86e059a41ec91ef690225b667e28e9ba5",
}
BASE_URL = "https://dl.fbaipublicfiles.com/LVIS/"


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    try:
        urlretrieve(url, partial)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def load_annotations(split, folder):
    archive = folder / f"lvis_v1_{split}.json.zip"
    if not archive.exists():
        print(f"Downloading official {split} annotations...")
        download(BASE_URL + archive.name, archive)
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    if checksum != ARCHIVES[split]:
        raise ValueError(f"{archive} does not match the official LVIS v1 archive")
    with ZipFile(archive) as zip_file:
        with zip_file.open(f"lvis_v1_{split}.json") as file:
            return json.load(file)


def yolo_box(box, width, height):
    x, y, w, h = map(float, box)  # COCO: top-left x/y, width, height
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width + 1e-5 or y + h > height + 1e-5:
        raise ValueError(f"Invalid LVIS plate box: {box}")
    return f"{YOLO_ID} {(x+w/2)/width:.9f} {(y+h/2)/height:.9f} {w/width:.9f} {h/height:.9f}"


def prepare_split(split, data, output, coco_dir=None):
    plates = defaultdict(list)
    for annotation in data["annotations"]:
        if annotation["category_id"] == PLATE_ID:
            plates[annotation["image_id"]].append(annotation["bbox"])

    images = {image["id"]: image for image in data["images"]}
    image_dir = output / "images"
    label_dir = output / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    count, boxes, flagged, failures, review_rows = 0, 0, 0, [], []
    for image_id, rectangles in sorted(plates.items()):
        info = images[image_id]
        image_path = image_dir / f"{split}__{image_id}.jpg"
        label_path = label_dir / f"{split}__{image_id}.txt"
        url = info["coco_url"].replace("http://images.cocodataset.org/", "https://images.cocodataset.org/")
        try:
            if not image_path.exists():
                # Reuse local COCO photos if already downloaded.
                source = coco_dir / "/".join(url.split("/")[-2:]) if coco_dir else None
                if source and source.is_file():
                    shutil.copy2(source, image_path)
                else:
                    download(url, image_path)
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                if image.size != (info["width"], info["height"]):
                    raise ValueError("COCO photo dimensions disagree with LVIS")
            lines = [yolo_box(box, info["width"], info["height"]) for box in rectangles]
            content = "\n".join(lines) + "\n"
            if label_path.exists() and label_path.read_text() != content:
                raise ValueError("existing label differs; refusing to overwrite")
            label_path.write_text(content)
        except (OSError, ValueError) as error:
            failures.append(f"{split} {image_id}: {error}")
            if not label_path.exists():
                image_path.unlink(missing_ok=True)  # retry damaged/incomplete downloads
            continue
        count += 1
        boxes += len(rectangles)
        incomplete = PLATE_ID in info.get("not_exhaustive_category_ids", [])
        flagged += incomplete
        review_rows.append([image_path.name, split, url, info.get("license", ""), len(rectangles), incomplete])
    print(f"{split}: {count} photos, {boxes} plate boxes, {flagged} incomplete-plate-label flags")
    return failures, review_rows


def import_archive(archive_path):
    """Add LVIS photos locally: 344 provisional plates, 44 held for review."""
    from zipfile import ZipFile

    kept, held, kept_rows, held_rows = 0, 0, [], []
    with ZipFile(archive_path) as zip_file:
        prefix = "LVIS_Paper_Plate_Candidate/"
        manifest = json.loads(zip_file.read(prefix + "manifest.json"))
        for item in manifest:
            source_image = prefix + item["image_path"]
            source_label = prefix + item["label_path"]
            image_bytes = zip_file.read(source_image)
            label_bytes = zip_file.read(source_label)
            if hashlib.sha256(image_bytes).hexdigest() != item["sha256"]:
                raise ValueError(f"Image checksum mismatch: {source_image}")
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.verify()
            with Image.open(io.BytesIO(image_bytes)) as image:
                if image.size != (item["width"], item["height"]):
                    raise ValueError(f"Incorrect dimensions: {source_image}")
            if len(label_bytes.splitlines()) != item["instances"]:
                raise ValueError(f"Plate annotation count mismatch: {source_label}")
            for line in label_bytes.decode().splitlines():
                parts = line.split()
                if len(parts) != 5 or int(parts[0]) != YOLO_ID:
                    raise ValueError(f"Invalid plate label: {source_label}")
                x, y, width, height = map(float, parts[1:])
                if not (width > 0 and height > 0 and
                        x-width/2 >= -1e-6 and y-height/2 >= -1e-6 and
                        x+width/2 <= 1+1e-6 and y+height/2 <= 1+1e-6):
                    raise ValueError(f"Invalid plate geometry: {source_label}")
            flagged = bool(item["non_exhaustive_plate"])
            destination = (ROOT / "data/candidates/plate" if flagged
                           else ROOT / "data/processed/plate")
            filename = f"{item['split']}__{Path(source_image).name}"
            image_path = destination / "images" / filename
            label_path = destination / "labels" / (Path(filename).stem + ".txt")
            for path, payload in ((image_path, image_bytes), (label_path, label_bytes)):
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() and path.read_bytes() != payload:
                    raise ValueError(f"Refusing to overwrite different file: {path}")
                if not path.exists():
                    path.write_bytes(payload)
            record = dict(source_split=item["split"], source_image=Path(source_image).name,
                          processed_image=filename, processed_label=label_path.name,
                          instances_kept=item["instances"], sha256=item["sha256"],
                          source_class="LVIS paper_plate (756)", target_class_id=YOLO_ID,
                          suggested_split_group=f"lvis_{item['image_id']}",
                          review_status="flagged_hold" if flagged else "provisional",
                          image_license_id=item["license_id"], coco_url=item["coco_url"],
                          plate_labels_incomplete=flagged)
            (held_rows if flagged else kept_rows).append(record)
            if flagged:
                held += 1
            else:
                kept += 1
    for path, records in ((ROOT / "data/manifests/plate.csv", kept_rows),
                          (ROOT / "data/candidates/plate/flagged_for_review.csv", held_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(f"Added {kept} provisional plate photos; held {held} flagged photos for review.")
    print("Provisional labels may omit other cafeteria objects; NOT ready for final evaluation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-dir", type=Path, help="Optional directory holding existing COCO train2017/val2017 photos")
    parser.add_argument("--import-archive", type=Path, help="Import the verified LVIS candidate ZIP locally")
    args = parser.parse_args()
    if args.import_archive:
        import_archive(args.import_archive)
        return
    archives = ROOT / "data/raw/lvis"
    output = ROOT / "data/candidates/plate"
    failures, review_rows = [], []
    for split in ("train", "val"):
        data = load_annotations(split, archives)
        failed, rows = prepare_split(split, data, output, args.coco_dir)
        failures.extend(failed)
        review_rows.extend(rows)
    with (output / "review.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "lvis_split", "coco_url", "image_license_id", "plate_boxes", "plate_labels_incomplete"])
        writer.writerows(review_rows)
    if failures:
        for failure in failures:
            print("FAILED:", failure)
        raise SystemExit(f"{len(failures)} files failed; rerun to retry downloads")
    print(f"Candidate photos/labels saved to {output}")
    print("Review all visible cafeteria classes BEFORE promoting any image to data/processed/.")


if __name__ == "__main__":
    main()
