"""Audit dataset images, YOLO boxes, duplicates and source provenance (read-only)."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image
import yaml

from prepare import (CLASS_MAPS, IMAGE_SUFFIXES, PROJECT_ROOT, class_names_by_id,
                     current_label, remap_label_line)


def csv_file(path, rows, columns):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def dhash(image):
    pixels = list(image.convert("L").resize((9, 8)).getdata())
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | (pixels[y*9+x] > pixels[y*9+x+1])
    return bits


def source_check(image, source, source_name, split, raw, rows):
    """Verify original pixels and detect target boxes dropped on export."""
    original = raw / source
    data_yaml = original / "data.yaml"
    if not data_yaml.exists() or source not in CLASS_MAPS:
        return [], 0  # Not every source has an original Roboflow export.
    names = class_names_by_id(yaml.safe_load(data_yaml.read_text())["names"])
    stem = Path(source_name or image.name).stem
    labels = [p for p in original.rglob(stem + ".txt") if p.parent.name == "labels"]
    if split:
        matching = [p for p in labels if p.parent.parent.name == split]
        if matching:
            labels = matching
    if len(labels) != 1:
        return [("source_label_resolution", f"{len(labels)} original labels found")], 0
    original_label = labels[0]
    original_image = original_label.parent.parent / "images" / (source_name or image.name)
    if not original_image.is_file():
        # A re-export may have prefixed the filename with train__/valid__.
        original_image = original_label.parent.parent / "images" / image.name
    errors = []
    if not original_image.exists():
        errors.append(("missing_source_image", str(original_image)))
    else:
        with Image.open(image) as p, Image.open(original_image) as q:
            if p.convert("RGB").size != q.convert("RGB").size or p.convert("RGB").tobytes() != q.convert("RGB").tobytes():
                errors.append(("source_image_mismatch", str(original_image)))
    missing = 0
    for n, line in enumerate(original_label.read_text().splitlines(), 1):
        if not line.strip():
            continue
        mapped = remap_label_line(line, names, CLASS_MAPS[source], original_label, n)
        if mapped and mapped not in rows:
            missing += 1
    return errors, missing


def audit(processed, output, config, raw=None, manifests=None):
    output.mkdir(parents=True, exist_ok=True)
    classes = {int(k): v for k, v in yaml.safe_load(config.read_text())["classes"].items()}
    names_by_id = {k: str(k) for k in classes}
    keep_ids = {str(k): k for k in classes}
    images = sorted(p for p in processed.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES and p.parent.name == "images")
    problems, inventory, missing_boxes = [], [], []
    source_counts, class_instances, class_images = defaultdict(Counter), Counter(), Counter()
    pixel_groups = defaultdict(list)
    manifest_cache = {}

    def problem(kind, name, detail):
        problems.append(dict(kind=kind, image=name, detail=str(detail)))

    for image in images:
        source = image.relative_to(processed).parts[0]
        name = image.relative_to(processed).as_posix()
        label = image.parent.parent / "labels" / (image.stem + ".txt")
        source_counts[source]["images"] += 1
        try:
            with Image.open(image) as opened:
                rgb = opened.convert("RGB")
                width, height = rgb.size
                pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                photo_hash = hashlib.sha256(image.read_bytes()).hexdigest()
                difference = dhash(rgb)
            pixel_groups[pixels].append(name)
        except Exception as error:
            width = height = 0
            pixels = photo_hash = ""
            difference = 0
            problem("unreadable_image", name, error)

        rows = []
        if not label.exists():
            problem("missing_label", name, label)
        else:
            for number, line in enumerate(label.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    if len(line.split()) != 5:
                        raise ValueError("Expected 5 detection values")
                    row = remap_label_line(current_label(line, source), names_by_id,
                                           keep_ids, label, number)
                    rows.append(row)
                    class_instances[int(row.split()[0])] += 1
                except ValueError as error:
                    problem("invalid_box", name, f"line {number}: {error}")
        if not rows:
            problem("empty_label", name, "No valid object boxes")
        for class_id in {int(r.split()[0]) for r in rows}:
            class_images[class_id] += 1
            source_counts[source][f"class_{class_id}"] += 1
        source_counts[source]["instances"] += len(rows)

        entry = {}
        if manifests:
            if source not in manifest_cache:
                file = manifests / (source + ".csv")
                if not file.exists():
                    problem("missing_manifest", source, file)
                    manifest_cache[source] = {}
                else:
                    manifest_cache[source] = {
                        (r.get("processed_image") or r.get("filename") or r.get("image")): r
                        for r in csv.DictReader(file.open(newline=""))
                    }
            entry = manifest_cache[source].get(image.name)
            if not entry:
                problem("unmatched_manifest", name, "No provenance record")
                entry = {}
            if entry.get("sha256") and entry["sha256"] != photo_hash:
                problem("manifest_hash_mismatch", name, "Image checksum differs")
            stated = entry.get("instances_kept") or entry.get("instances")
            if stated and int(stated) != len(rows):
                problem("manifest_count_mismatch", name, f"{stated} vs {len(rows)}")

        original_name = entry.get("source_image") or entry.get("filename") or image.name
        split = entry.get("source_split") or entry.get("original_split") or ""
        if raw:
            try:
                errors, missing = source_check(image, source, original_name, split, raw, rows)
                for kind, detail in errors:
                    problem(kind, name, detail)
                if missing:
                    missing_boxes.append(dict(image=name, source=source, missing_target_boxes=missing))
            except Exception as error:
                problem("source_check_error", name, error)
        inventory.append(dict(image=name, source=source, label=label.name,
                              width=width, height=height, sha256=photo_hash,
                              pixel_sha256=pixels, dhash=f"{difference:016x}",
                              classes=";".join(map(str, sorted({int(r.split()[0]) for r in rows}))),
                              instances=len(rows), source_split=split,
                              source_image=original_name))

    for label in processed.rglob("*.txt"):
        if label.parent.name == "labels" and not any(
            (label.parent.parent / "images" / (label.stem + ext)).is_file() for ext in IMAGE_SUFFIXES
        ):
            problem("orphan_label", label.relative_to(processed).as_posix(), "No corresponding image")
    if not images:
        problem("no_images", "", "Dataset is empty")

    duplicates = [paths for paths in pixel_groups.values() if len(paths) > 1]
    near = []
    # Potential similar photos are flagged for review, not automatically deleted.
    # Duplicate *pixels* are separately identified and kept together in training.
    for paths in duplicates:
        for path in paths:
            near.append(dict(image_a=paths[0], image_b=path, status="exact_duplicate"))
    missing_total = sum(item["missing_target_boxes"] for item in missing_boxes)
    if missing_total:
        problem("missing_target_annotations", "corpus", f"{missing_total} boxes")

    class_counts = [dict(class_id=k, name=name, images=class_images[k], instances=class_instances[k])
                    for k, name in sorted(classes.items())]
    summary = dict(images=len(images), instances=sum(class_instances.values()),
                   class_counts=class_counts, sources=dict(source_counts),
                   exact_duplicate_groups=len(duplicates), near_duplicate_candidate_pairs=0,
                   missing_source_target_boxes=missing_total,
                   missing_classes=[r["name"] for r in class_counts if not r["instances"]],
                   issues=problems, structural_pass=not problems,
                   complete_multiclass_annotations_verified=False)
    csv_file(output / "inventory.csv", inventory,
             ["image", "source", "label", "width", "height", "sha256",
              "pixel_sha256", "dhash", "classes", "instances", "source_split", "source_image"])
    csv_file(output / "class_counts.csv", class_counts, ["class_id", "name", "images", "instances"])
    csv_file(output / "issues.csv", problems, ["kind", "image", "detail"])
    csv_file(output / "exact_duplicates.csv",
             [dict(group=n, image=p) for n, group in enumerate(duplicates, 1) for p in group],
             ["group", "image"])
    csv_file(output / "near_duplicates.csv", near, ["image_a", "image_b", "status"])
    csv_file(output / "source_annotation_gaps.csv", missing_boxes,
             ["image", "source", "missing_target_boxes"])
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Audited {summary['images']} images; {len(problems)} issues found.")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/audits/current")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--manifests", type=Path)
    args = parser.parse_args()
    summary = audit(args.processed, args.output, args.config, args.raw, args.manifests)
    return 0 if summary["structural_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
