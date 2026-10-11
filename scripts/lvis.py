"""Download or import LVIS paper-plate photos as review candidates, not final test data."""
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
PLATE_ID = 756  # LVIS v1 "paper plate"
YOLO_ID = 1
BASE_URL = "https://dl.fbaipublicfiles.com/LVIS/"
ARCHIVE_HASHES = {
    "train": "334a4caa374030a7817cf050364525e910f7960f9b6968cef47cffbf3893f8ba",
    "val": "5cae9a3c79aadb667550c2b5dcf7f4d86e059a41ec91ef690225b667e28e9ba5",
}


def download(url, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    try:
        urlretrieve(url, partial)
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)


def load_annotations(split, folder):
    archive = folder / f"lvis_v1_{split}.json.zip"
    if not archive.exists():
        download(BASE_URL + archive.name, archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_HASHES[split]:
        raise ValueError(f"Invalid official LVIS annotation archive: {archive}")
    with ZipFile(archive) as z:
        return json.loads(z.read(f"lvis_v1_{split}.json"))


def yolo_box(box, width, height):
    x, y, w, h = map(float, box)
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x+w > width+1e-5 or y+h > height+1e-5:
        raise ValueError("LVIS box lies outside the image")
    return f"{YOLO_ID} {(x+w/2)/width:.9f} {(y+h/2)/height:.9f} {w/width:.9f} {h/height:.9f}"


def prepare_split(split, data, output, coco_dir=None):
    """Write LVIS plate images and YOLO boxes for review."""
    boxes = defaultdict(list)
    for annotation in data["annotations"]:
        if annotation["category_id"] == PLATE_ID:
            boxes[annotation["image_id"]].append(annotation["bbox"])
    catalog = {image["id"]: image for image in data["images"]}
    images, labels = output / "images", output / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    failures, review = [], []
    for image_id, rectangles in sorted(boxes.items()):
        info = catalog[image_id]
        url = info["coco_url"].replace("http://images.cocodataset.org/",
                                       "https://images.cocodataset.org/")
        image_file = images / f"{split}__{image_id}.jpg"
        label_file = labels / f"{split}__{image_id}.txt"
        try:
            if not image_file.exists():
                local = (coco_dir / "/".join(url.split("/")[-2:])) if coco_dir else None
                if local and local.is_file():
                    shutil.copy2(local, image_file)
                else:
                    download(url, image_file)
            with Image.open(image_file) as image:
                image.verify()
            with Image.open(image_file) as image:
                if image.size != (info["width"], info["height"]):
                    raise ValueError("Image dimensions disagree with LVIS")
            lines = [yolo_box(box, info["width"], info["height"]) for box in rectangles]
            text = "\n".join(lines) + "\n"
            if label_file.exists() and label_file.read_text() != text:
                raise ValueError("Existing label does not match official LVIS")
            label_file.write_text(text)
        except (OSError, ValueError) as exc:
            failures.append(f"{split} {image_id}: {exc}")
            if not label_file.exists():
                image_file.unlink(missing_ok=True)
            continue
        review.append([image_file.name, split, url, info.get("license", ""), len(rectangles),
                       PLATE_ID in info.get("not_exhaustive_category_ids", [])])
    return failures, review


def import_archive(archive):
    """Import verified candidate ZIP: 344 provisional, 44 flagged/held."""
    output = {"accepted": [], "held": []}
    with ZipFile(archive) as z:
        items = json.loads(z.read("LVIS_Paper_Plate_Candidate/manifest.json"))
        for item in items:
            prefix = "LVIS_Paper_Plate_Candidate/"
            picture = z.read(prefix + item["image_path"])
            annotations = z.read(prefix + item["label_path"])
            if hashlib.sha256(picture).hexdigest() != item["sha256"]:
                raise ValueError("LVIS image SHA256 mismatch")
            with Image.open(io.BytesIO(picture)) as image:
                image.verify()
            with Image.open(io.BytesIO(picture)) as image:
                if image.size != (item["width"], item["height"]):
                    raise ValueError("Wrong LVIS image dimensions")
            lines = annotations.decode().splitlines()
            if len(lines) != item["instances"]:
                raise ValueError("LVIS annotation count mismatch")
            for line in lines:
                parts = line.split()
                if len(parts) != 5 or int(parts[0]) != YOLO_ID:
                    raise ValueError("Invalid LVIS class annotation")
                x, y, w, h = map(float, parts[1:])
                if w <= 0 or h <= 0 or min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6:
                    raise ValueError("Invalid LVIS bounding box")
            held = bool(item["non_exhaustive_plate"])
            folder = ROOT / ("data/candidates/plate" if held else "data/processed/plate")
            name = f'{item["split"]}__{Path(item["image_path"]).name}'
            for file, payload in ((folder / "images" / name, picture),
                                  (folder / "labels" / (Path(name).stem + ".txt"), annotations)):
                file.parent.mkdir(parents=True, exist_ok=True)
                if file.exists() and file.read_bytes() != payload:
                    raise ValueError(f"Refusing to overwrite: {file}")
                if not file.exists():
                    file.write_bytes(payload)
            output["held" if held else "accepted"].append({
                "source_split": item["split"],
                "source_image": Path(item["image_path"]).name,
                "processed_image": name,
                "processed_label": Path(name).stem + ".txt",
                "instances_kept": item["instances"],
                "sha256": item["sha256"],
                "source_class": "LVIS paper_plate (756)",
                "target_class_id": YOLO_ID,
                "suggested_split_group": f'lvis_{item["image_id"]}',
                "review_status": "flagged_hold" if held else "provisional",
                "image_license_id": item["license_id"],
                "coco_url": item["coco_url"],
                "plate_labels_incomplete": held,
            })
    for dest, records in ((ROOT / "data/manifests/plate.csv", output["accepted"]),
                          (ROOT / "data/candidates/plate/flagged_for_review.csv", output["held"])):
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=records[0])
            writer.writeheader()
            writer.writerows(records)
    print(f"Imported {len(output['accepted'])} provisional plates; "
          f"held {len(output['held'])} incomplete examples for review")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-dir", type=Path)
    parser.add_argument("--import-archive", type=Path)
    args = parser.parse_args()
    if args.import_archive:
        import_archive(args.import_archive)
        return
    output = ROOT / "data/candidates/plate"
    output.mkdir(parents=True, exist_ok=True)
    errors, review = [], []
    for split in ("train", "val"):
        failed, rows = prepare_split(split, load_annotations(split, ROOT / "data/raw/lvis"),
                                     output, args.coco_dir)
        errors.extend(failed)
        review.extend(rows)
    with (output / "review.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "lvis_split", "coco_url", "image_license_id",
                         "plate_boxes", "plate_labels_incomplete"])
        writer.writerows(review)
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Saved {len(review)} plate candidate images for visual review")


if __name__ == "__main__":
    main()
