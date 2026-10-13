"""Prepare additive label corrections and bounded training views on a frozen split."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

import yaml
from PIL import Image

from prepare import IMAGE_SUFFIXES, PROJECT_ROOT, remap_label_line
from train import check_names, class_names


def validate_rows(rows, location, names):
    """Validate current IDs directly; frozen ground truth is already remapped."""
    by_id = dict(enumerate(names))
    by_name = {name: index for index, name in enumerate(names)}
    result = []
    for number, row in enumerate(rows, 1):
        if not isinstance(row, str) or len(row.split()) != 5:
            raise ValueError(f"{location}:{number}: expected one YOLO box with five numbers")
        result.append(remap_label_line(row, by_id, by_name, location, number))
    return result


def box_iou(first, second):
    """Overlap of two validated YOLO boxes, without altering either box."""
    a, b = [list(map(float, row.split()[1:])) for row in (first, second)]
    left = max(a[0] - a[2] / 2, b[0] - b[2] / 2)
    right = min(a[0] + a[2] / 2, b[0] + b[2] / 2)
    top = max(a[1] - a[3] / 2, b[1] - b[3] / 2)
    bottom = min(a[1] + a[3] / 2, b[1] + b[3] / 2)
    intersection = max(0, right - left) * max(0, bottom - top)
    return intersection / (a[2] * a[3] + b[2] * b[3] - intersection)


def check_output(output, sources):
    resolved = output.resolve()
    protected = [PROJECT_ROOT / "data/raw", PROJECT_ROOT / "data/processed",
                 PROJECT_ROOT / "model", *sources]
    if output.exists() or output.is_symlink():
        raise ValueError(f"Experiment output already exists; choose a new directory: {output}")
    if any(resolved == source.resolve() or resolved in source.resolve().parents
           or source.resolve() in resolved.parents for source in protected):
        raise ValueError(f"Experiment output overlaps source or protected data: {output}")


def label_path(image):
    parts = list(image.parts)
    positions = [index for index, part in enumerate(parts) if part == "images"]
    if not positions:
        raise ValueError(f"Source image must be in an images directory: {image}")
    parts[positions[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def class_support(samples, names, repeated=False, original=False):
    result = {}
    for split in dict.fromkeys(sample["split"] for sample in samples):
        images, instances = Counter(), Counter()
        for sample in samples:
            if sample["split"] != split:
                continue
            count = sample["repeat"] if repeated else 1
            rows = sample["original_rows"] if original else sample["rows"]
            ids = [int(row.split()[0]) for row in rows]
            images.update({index: count for index in set(ids)})
            instances.update({index: ids.count(index) * count for index in set(ids)})
        result[split] = {name: {"images": images[index], "instances": instances[index]}
                         for index, name in enumerate(names)}
    return result


def apply_additions(samples, additions_file, names):
    report = {"path": None, "sha256": None, "requested": 0, "accepted": 0,
              "duplicates_skipped": 0, "by_split": {sample["split"]: 0 for sample in samples},
              "records": []}
    if additions_file is None:
        return report
    additions_file = Path(additions_file)
    raw = additions_file.read_bytes()
    records = json.loads(raw)
    if not isinstance(records, list):
        raise ValueError("Label additions must be a JSON list of records")
    report.update(path=str(additions_file.resolve()), sha256=hashlib.sha256(raw).hexdigest())
    by_stem = {}
    for sample in samples:
        by_stem.setdefault(sample["image"].stem, []).append(sample)
    for number, record in enumerate(records, 1):
        location = f"{additions_file}:record {number}"
        if not isinstance(record, dict):
            raise ValueError(f"{location}: expected an addition record")
        source, image = record.get("source"), record.get("image")
        if (not isinstance(source, str) or not source or source in {".", ".."}
                or "/" in source or "\\" in source or not isinstance(image, str)
                or Path(image).name != image or "/" in image or "\\" in image
                or Path(image).suffix.lower() not in IMAGE_SUFFIXES):
            raise ValueError(f"{location}: expected source and original image filename")
        matches = by_stem.get(source + "__" + Path(image).stem, [])
        if len(matches) != 1:
            raise ValueError(f"{location}: unmatched or ambiguous original image {source}/{image}")
        if not isinstance(record.get("rows"), list) or not record.get("provenance"):
            raise ValueError(f"{location}: rows and provenance are required")
        sample = matches[0]
        accepted, skipped = [], []
        for row in validate_rows(record["rows"], location, names):
            report["requested"] += 1
            if any(row.split()[0] == existing.split()[0] and box_iou(row, existing) > 0.85
                   for existing in sample["rows"]):
                skipped.append(row)
                report["duplicates_skipped"] += 1
                continue
            sample["rows"].append(row)
            sample["added_rows"].append(row)
            accepted.append(row)
            report["accepted"] += 1
            report["by_split"][sample["split"]] += 1
        report["records"].append({"source": source, "image": image, "split": sample["split"],
                                  "accepted_rows": accepted, "duplicate_rows": skipped,
                                  "provenance": record["provenance"]})
    return report


def focus_crop(image, rows, names, focus_classes, view, crop_fraction=0.60,
               small_area=0.04):
    """Make a label-preserving crop around a genuinely small weak-class box.

    Returns None if no safe crop target exists. Other partially visible
    labeled objects are clipped and carried forward, not silently erased.
    """
    wanted = {names.index(name) for name in focus_classes}
    candidates = []
    for row in rows:
        cls, x, y, w, h = map(float, row.split())
        if (int(cls) in wanted and w * h < small_area
                and w <= crop_fraction * 0.95 and h <= crop_fraction * 0.95):
            candidates.append((w * h, int(cls), x, y, w, h))
    if not candidates:
        return None
    candidates.sort()
    _, _, x, y, _, _ = candidates[(view - 1) % len(candidates)]
    with Image.open(image) as picture:
        width, height = picture.size
        crop_w = max(2, round(width * crop_fraction))
        crop_h = max(2, round(height * crop_fraction))
        left = max(0, min(width - crop_w, round(x * width - crop_w / 2)))
        top = max(0, min(height - crop_h, round(y * height - crop_h / 2)))
        right, bottom = left + crop_w, top + crop_h
        transformed = []
        for row in rows:
            parts = row.split()
            cls = int(parts[0])
            bx, by, bw, bh = map(float, parts[1:])
            a = max(left, (bx - bw / 2) * width)
            b = max(top, (by - bh / 2) * height)
            c = min(right, (bx + bw / 2) * width)
            d = min(bottom, (by + bh / 2) * height)
            if c - a < 1 or d - b < 1:
                continue
            transformed.append(
                f"{cls} {((a + c) / 2 - left) / crop_w:.9f} "
                f"{((b + d) / 2 - top) / crop_h:.9f} "
                f"{(c - a) / crop_w:.9f} {(d - b) / crop_h:.9f}")
        if not transformed:
            return None
        return picture.crop((left, top, right, bottom)).convert("RGB"), transformed


def prepare_experiment(source, output, label_additions=None, balance_classes=(),
                       max_repeat=3, crop_repeats=False):
    """Create a new dataset; retain all fixed source images and every original box."""
    source, output = Path(source).resolve(), Path(output)
    source_yaml = source if source.suffix in {".yaml", ".yml"} else source / "data.yaml"
    raw_yaml = source_yaml.read_bytes()
    data = yaml.safe_load(raw_yaml)
    names = class_names()
    check_names(data["names"], "source data.yaml")
    data_root = Path(data.get("path") or source_yaml.parent)
    if not data_root.is_absolute():
        data_root = source_yaml.parent / data_root
    check_output(output, [source_yaml.parent, data_root])
    classes = list(dict.fromkeys(balance_classes))
    if set(classes) - set(names):
        raise ValueError("Unknown balancing classes: " + ", ".join(sorted(set(classes) - set(names))))
    if isinstance(max_repeat, bool) or not isinstance(max_repeat, int) or not 1 <= max_repeat <= 3:
        raise ValueError("max-repeat must be an integer between 1 and 3")
    if crop_repeats and (not classes or max_repeat < 2):
        raise ValueError("--crop-repeats needs weak classes and max-repeat at least 2")

    samples = []
    splits = ["train", "val"] + (["test"] if data.get("test") else [])
    for split in splits:
        directory = Path(data[split])
        if not directory.is_absolute():
            directory = data_root / directory
        if not directory.is_dir():
            raise ValueError(f"Source {split} must identify an existing images directory: {directory}")
        check_output(output, [directory])
        images = sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise ValueError(f"No original images in source {split}: {directory}")
        for image in images:
            label = label_path(image)
            check_output(output, [label.parent])
            raw_label = label.read_bytes()
            rows = validate_rows([row for row in raw_label.decode().splitlines() if row.strip()], label, names)
            samples.append({"split": split, "image": image, "label": label, "raw_label": raw_label,
                            "original_rows": rows[:], "rows": rows, "added_rows": [], "repeat": 1})

    additions = apply_additions(samples, label_additions, names)
    corrected_support = class_support(samples, names)
    target = max(row["images"] for row in corrected_support["train"].values())
    factors = {name: min(max_repeat, math.ceil(target / corrected_support["train"][name]["images"]))
               if corrected_support["train"][name]["images"] else 1 for name in classes}
    names_by_id = dict(enumerate(names))
    destination_names = set()
    for sample in samples:
        if sample["split"] == "train":
            sample["repeat"] = max([1] + [factors.get(names_by_id[int(row.split()[0])], 1)
                                          for row in sample["rows"]])
        for view in range(sample["repeat"]):
            stem = sample["image"].stem + (f"__balanced{view:02d}" if view else "")
            key = (sample["split"], stem)
            if key in destination_names:
                raise ValueError(f"Generated view collides with an original filename: {stem}")
            destination_names.add(key)

    # Complete validation before creating the new output; existing directories
    # are never removed, and original labels are never linked to writable copies.
    output.mkdir(parents=True)
    cropped_views = 0
    actual_views = []
    for sample in samples:
        label_bytes = sample["raw_label"]
        if sample["added_rows"]:
            separator = b"\n" if label_bytes and not label_bytes.endswith(b"\n") else b""
            label_bytes += separator + ("\n".join(sample["added_rows"]) + "\n").encode()
        for view in range(sample["repeat"]):
            stem = sample["image"].stem + (f"__balanced{view:02d}" if view else "")
            image_dir = output / "images" / sample["split"]
            labels_dir = output / "labels" / sample["split"]
            image_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            target_image = image_dir / (stem + sample["image"].suffix)
            crop = (focus_crop(sample["image"], sample["rows"], names, classes, view)
                    if crop_repeats and sample["split"] == "train" and view else None)
            if crop is None:
                try:
                    os.link(sample["image"].resolve(), target_image)
                except OSError:
                    shutil.copy2(sample["image"], target_image)
                (labels_dir / (stem + ".txt")).write_bytes(label_bytes)
                view_rows = sample["rows"]
            else:
                picture, cropped_rows = crop
                # Lossless PNG keeps the small target visible; extension must
                # reflect actual encoded bytes for the image loader.
                target_image = target_image.with_suffix(".png")
                if target_image.exists():
                    raise ValueError(f"Crop output collision: {target_image}")
                picture.save(target_image, format="PNG")
                (labels_dir / (stem + ".txt")).write_text(
                    "\n".join(validate_rows(cropped_rows, target_image, names)) + "\n")
                cropped_views += 1
                view_rows = cropped_rows
            actual_views.append({"split": sample["split"], "rows": view_rows,
                                 "repeat": 1})

    data_file = output / "data.yaml"
    data_file.write_text(yaml.safe_dump({"path": str(output.resolve()), "names": names,
                                       **{split: str((output / "images" / split).resolve())
                                          for split in splits}}, sort_keys=False))
    counts = {key: value for split in splits for key, value in (
        (split + "_originals", sum(sample["split"] == split for sample in samples)),
        (split + "_views", sum(sample["repeat"] for sample in samples if sample["split"] == split)))}
    report = {
        "source_yaml": str(source_yaml.resolve()), "source_yaml_sha256": hashlib.sha256(raw_yaml).hexdigest(),
        "output": str(output.resolve()), "counts": counts, "additions": additions,
        "balance": {"classes": classes, "target_image_count": target, "max_repeat": max_repeat,
                    "class_repeat_factors": factors},
        "focus_crops": {"enabled": crop_repeats, "cropped_train_views": cropped_views,
                        "crop_fraction": 0.60, "max_target_area": 0.04,
                        "validation_unchanged": True},
        "original_support": class_support(samples, names, original=True),
        "corrected_support": corrected_support, "view_support": class_support(actual_views, names),
        "source_image_paths": {split: sorted(str(sample["image"].resolve()) for sample in samples
                                              if sample["split"] == split) for split in splits},
        "evaluation_cohorts": {split: {name: sorted(str(sample["image"].resolve()) for sample in samples
                                                    if sample["split"] == split
                                                    and index in {int(row.split()[0]) for row in sample["rows"]})
                                        for index, name in enumerate(names)} for split in splits if split != "train"},
        "original_label_sha256": {str(sample["label"].resolve()): hashlib.sha256(sample["raw_label"]).hexdigest()
                                  for sample in samples},
    }
    (output / "experiment_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Frozen split: {counts['train_originals']} training originals, {counts['train_views']} views; "
          f"{counts['val_originals']} validation originals, {counts['val_views']} views")
    print(f"Accepted {additions['accepted']} additional boxes; skipped {additions['duplicates_skipped']} overlaps")
    print("Prepared experiment:", data_file)
    return data_file



def prepare_weak_subset(source, output, weak_classes=("plate", "wrapper", "bottle"),
                        replay_fraction=0.15, seed=42):
    """Make small train-only manifest; preserve ALL 11 labels and the full validation.

    Original and focus-crop views of one source photo are selected together.
    Strong-only replay guards against catastrophic forgetting during fine-tuning.
    This operation creates just train.txt, data.yaml and a short JSON report.
    """
    source = Path(source).resolve()
    source_yaml = source if source.suffix in {".yaml", ".yml"} else source / "data.yaml"
    data = yaml.safe_load(source_yaml.read_text())
    names = class_names()
    check_names(data["names"], "source data.yaml")
    weak_classes = tuple(dict.fromkeys(weak_classes))
    if not weak_classes or set(weak_classes) - set(names):
        raise ValueError("Weak classes must be nonempty valid class names")
    if (isinstance(replay_fraction, bool) or
            not isinstance(replay_fraction, (int, float)) or
            not math.isfinite(replay_fraction) or not 0 <= replay_fraction <= 1):
        raise ValueError("Replay fraction must be between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Seed must be an integer")
    root = Path(data.get("path") or source_yaml.parent)
    if not root.is_absolute():
        root = source_yaml.parent / root
    train_dir = Path(data["train"])
    val_dir = Path(data["val"])
    if not train_dir.is_absolute():
        train_dir = root / train_dir
    if not val_dir.is_absolute():
        val_dir = root / val_dir
    train_dir, val_dir = train_dir.resolve(), val_dir.resolve()
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise ValueError("Weak subset needs existing train/validation image directories")
    output = Path(output).resolve()
    check_output(output, [root, source_yaml.parent, train_dir, val_dir])
    images = sorted(p for p in train_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError("Source has no training photos")
    groups = {}
    for image in images:
        label = label_path(image)
        if not label.is_file():
            raise ValueError("Missing training label: " + str(label))
        rows = validate_rows([r for r in label.read_text().splitlines() if r.strip()],
                             label, names)
        # Keep original plus every __balanced view in the SAME group.
        key = __import__("re").sub(r"__balanced\\d+$", "", image.stem)
        group = groups.setdefault(key, {"images": [], "class_ids": set()})
        group["images"].append(image)
        group["class_ids"].update(int(row.split()[0]) for row in rows)
    selected_ids = {names.index(name) for name in weak_classes}
    selected_weak = {key for key, group in groups.items()
                     if group["class_ids"] & selected_ids}
    if not selected_weak:
        raise ValueError("No training images contain selected weak classes")
    replay_keys = sorted(
        (key for key in groups if key not in selected_weak),
        key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).digest())
    replay_n = math.ceil(len(replay_keys) * replay_fraction)
    selected = selected_weak | set(replay_keys[:replay_n])
    chosen = sorted(image.resolve()
                    for key in selected for image in groups[key]["images"])
    support = Counter()
    for key in selected:
        support.update(groups[key]["class_ids"])
    if any(support[cls] == 0 for cls in selected_ids):
        raise ValueError("Weak subset lost a required class")
    output.mkdir(parents=True)
    manifest = output / "train.txt"
    manifest.write_text("".join(str(image) + "\\n" for image in chosen))
    result = dict(data)
    result["path"] = str(root.resolve())
    result["train"] = str(manifest.resolve())
    result["val"] = str(val_dir)
    # Keep the existing public validation and independent test, if any, untouched.
    if data.get("test"):
        test_dir = Path(data["test"])
        result["test"] = str((test_dir if test_dir.is_absolute()
                              else root / test_dir).resolve())
    (output / "data.yaml").write_text(yaml.safe_dump(result, sort_keys=False))
    details = {
        "weak_classes": list(weak_classes), "replay_fraction": replay_fraction,
        "seed": seed, "source": str(source_yaml), "source_training_views": len(images),
        "selected_training_views": len(chosen),
        "weak_original_groups": len(selected_weak),
        "strong_only_groups_in_source": len(replay_keys),
        "replay_original_groups": replay_n,
        "selected_class_group_support": {name: support[i] for i, name in enumerate(names)},
        "validation_path": str(val_dir),
        "validation_unchanged": True,
        "all_class_labels_preserved": True,
    }
    (output / "weak_subset.json").write_text(json.dumps(details, indent=2) + "\\n")
    print(f"Weak-class fast subset: {len(chosen)} of {len(images)} training views "
          f"({len(chosen) / len(images):.1%}).")
    print(f"Selected {len(selected_weak)} weak original groups + "
          f"{replay_n} strong-class replay groups. Validation unchanged.")
    print("Prepared subset:", output / "data.yaml")
    return output / "data.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "data/training")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/training-revised")
    parser.add_argument("--label-additions", type=Path, help="Reviewed additive boxes and their provenance in JSON")
    parser.add_argument("--balance-classes", nargs="*", default=[], help="Classes receiving extra training views")
    parser.add_argument("--max-repeat", type=int, default=3, help="Maximum views per training original (1–3)")
    parser.add_argument("--crop-repeats", action="store_true",
                        help="Replace repeated train-only weak-class views with 60% focus crops")
    parser.add_argument("--weak-subset", action="store_true",
                        help="Create a fast weak-class training list without copying photos")
    parser.add_argument("--weak-classes", nargs="+",
                        default=["plate", "wrapper", "bottle"])
    parser.add_argument("--replay-fraction", type=float, default=0.15,
                        help="Fraction of strong-only original photos to retain (default 0.15)")
    args = parser.parse_args()
    try:
        if args.weak_subset:
            if args.label_additions or args.crop_repeats or args.balance_classes:
                raise ValueError("Weak-subset mode takes an existing prepared dataset; "
                                 "run label/crop experiments separately first")
            prepare_weak_subset(args.source, args.output,
                                args.weak_classes, args.replay_fraction)
        else:
            prepare_experiment(args.source, args.output, args.label_additions,
                               args.balance_classes, args.max_repeat, args.crop_repeats)
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
