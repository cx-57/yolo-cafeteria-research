"""Inspect a generated split and training curve; optionally prepare a source ablation.

Reads data/training images and current-ID labels without changing them. An ablation
filters training paths only, leaving validation images and labels exactly fixed.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

from PIL import Image
import yaml

from prepare import IMAGE_SUFFIXES, PROJECT_ROOT, class_names_by_id


def inspect_split(dataset, split, names, imgsz):
    images = sorted(p for p in (dataset / "images" / split).iterdir()
                    if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"No images in {dataset / 'images' / split}")
    boxes, class_images, small = Counter(), Counter(), Counter()
    sources = defaultdict(Counter)
    for image in images:
        source = image.name.split("__", 1)[0]
        sources[source]["images"] += 1
        with Image.open(image) as opened:
            width, height = opened.size
        scale = imgsz / max(width, height)
        label = dataset / "labels" / split / (image.stem + ".txt")
        ids = set()
        for number, row in enumerate(label.read_text().splitlines(), 1):
            if not row.strip():
                continue
            parts = row.split()
            if len(parts) != 5:
                raise ValueError(f"{label}:{number}: expected five detection values")
            cid = int(parts[0])
            coords = list(map(float, parts[1:]))
            if cid not in names or not all(math.isfinite(v) for v in coords):
                raise ValueError(f"{label}:{number}: invalid class or coordinates")
            x, y, w, h = coords
            if (w <= 0 or h <= 0 or x-w/2 < -1e-6 or y-h/2 < -1e-6
                    or x+w/2 > 1+1e-6 or y+h/2 > 1+1e-6):
                raise ValueError(f"{label}:{number}: box outside normalized image bounds")
            boxes[cid] += 1
            ids.add(cid)
            sources[source]["instances"] += 1
            if min(w * width * scale, h * height * scale) < 32:
                small[cid] += 1
        class_images.update(ids)
    return {
        "images": len(images), "instances": sum(boxes.values()),
        "sources": dict(sorted(sources.items())),
        "classes": {name: {"images": class_images[cid], "instances": boxes[cid],
                           "short_side_under_32px": small[cid]}
                    for cid, name in names.items()},
    }, images


def inspect_run(run):
    with (run / "results.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        rows = [{k.strip(): float(v) for k, v in row.items() if v and v.strip()}
                for row in reader]
    if not rows:
        raise ValueError(f"No completed epochs in {run / 'results.csv'}")
    if not all(math.isfinite(v) for row in rows for v in row.values()):
        raise ValueError(f"Nonfinite metrics in {run / 'results.csv'}")
    args = yaml.safe_load((run / "args.yaml").read_text()) or {}
    epochs = sorted({int(row["epoch"]) for row in rows})
    metrics = ("metrics/precision(B)", "metrics/recall(B)",
               "metrics/mAP50(B)", "metrics/mAP50-95(B)")
    best = {}
    for metric in metrics:
        available = [row for row in rows if metric in row]
        if available:
            row = max(available, key=lambda r: r[metric])
            best[metric] = {"epoch": int(row["epoch"]), "value": row[metric]}
    scheduled = args.get("epochs")
    close = args.get("close_mosaic", 0)
    return {
        "path": str(run.resolve()), "logged_epochs": len(rows),
        "last_epoch": epochs[-1], "planned_epochs": scheduled,
        "missing_epochs": sorted(set(range(1, epochs[-1] + 1)) - set(epochs)),
        "best": best, "last": rows[-1],
        "scheduled_mosaic_close_reached": bool(
            scheduled and close and epochs[-1] > scheduled - close),
        "stop_reason": "Not inferable from CSV; inspect training log/checkpoint.",
        "training_arguments": args,
    }


def inspect_saved_support(dataset, names):
    """Use saved image support when cloud-backed label reads are unavailable."""
    summary = json.loads((dataset / "split_summary.json").read_text())
    result, images = {}, {}
    for split in ("train", "val"):
        images[split] = sorted(p for p in (dataset / "images" / split).iterdir()
                               if p.suffix.lower() in IMAGE_SUFFIXES)
        count_key = "train_views" if split == "train" else "val_originals"
        if len(images[split]) != summary[count_key]:
            raise ValueError("Saved split summary image count is stale; run a fresh label scan")
        counts = summary[f"{split}_class_images_original"]
        originals = summary[f"{split}_originals"]
        if set(counts) != set(names.values()) or any(
                not isinstance(count, int) or isinstance(count, bool)
                or not 0 <= count <= originals for count in counts.values()):
            raise ValueError("Invalid saved class support")
        source_counts = Counter(p.name.split("__", 1)[0] for p in images[split])
        result[split] = {
            "images": len(images[split]), "instances": None,
            "class_image_basis": "originals_without_repeated_training_views",
            "sources": {source: {"images": count, "instances": None}
                        for source, count in sorted(source_counts.items())},
            "classes": {name: {"images": counts[name], "instances": None,
                               "short_side_under_32px": None} for name in names.values()},
        }
    return result, images


def write_ablation(dataset, output, data, images, excluded):
    available = {p.name.split("__", 1)[0] for p in images}
    unknown = set(excluded) - available
    if unknown:
        raise ValueError(f"Unknown training source: {', '.join(sorted(unknown))}")
    kept = [p for p in images if p.name.split("__", 1)[0] not in excluded]
    if not kept:
        raise ValueError("Exclusion would remove every training image")
    folder = output / "ablation"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "train.txt"
    manifest.write_text("".join(str(p.resolve()) + "\n" for p in kept))
    revised = dict(data)
    revised.update(path=str(dataset.resolve()), train=str(manifest.resolve()),
                   val=str((dataset / "images/val").resolve()))
    config = folder / "data.yaml"
    config.write_text(yaml.safe_dump(revised, sort_keys=False))
    return {"data_yaml": str(config.resolve()), "excluded_sources": sorted(excluded),
            "train_images_before": len(images), "train_images_after": len(kept),
            "validation_unchanged": True,
            "limitation": "Validation annotation issues remain; source ablation is diagnostic."}


def diagnose(dataset, output, run=None, imgsz=640, excluded=(), metadata_only=False):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output == dataset or dataset in output.parents:
        raise ValueError("Write diagnostic output outside the generated dataset")
    data = yaml.safe_load((dataset / "data.yaml").read_text())
    # Deliberately target the generated directory layout, rather than silently
    # inspect different images when given a custom YAML with lists or manifests.
    if data.get("train") != "images/train" or data.get("val") != "images/val":
        raise ValueError("Diagnosis requires the generated images/train and images/val layout")
    if "path" in data and Path(data["path"]).resolve() != dataset:
        raise ValueError("data.yaml path does not match --dataset")
    names = class_names_by_id(data["names"])
    splits, images = {}, {}
    if metadata_only:
        splits, images = inspect_saved_support(dataset, names)
    else:
        for split in ("train", "val"):
            splits[split], images[split] = inspect_split(dataset, split, names, imgsz)
    result = {
        "dataset": str(dataset), "imgsz": imgsz,
        "support_basis": ("saved_split_summary_not_fresh_label_scan" if metadata_only
                          else "fresh_generated_label_scan"),
        "small_box_definition": "Shorter side <32 pixels after aspect-preserving resize; heuristic, not COCO area metric.",
        "splits": splits,
        "missing_train_classes": [n for n in names.values()
                                  if not splits["train"]["classes"][n]["images"]],
        "missing_val_classes": [n for n in names.values()
                                if not splits["val"]["classes"][n]["images"]],
        "annotation_completeness": "Counts verify stored labels, not every visible target object.",
    }
    if run:
        result["run"] = inspect_run(Path(run))
    # Validate exclusion before writing any diagnostic artifact.
    if excluded:
        unknown = set(excluded) - {p.name.split("__", 1)[0] for p in images["train"]}
        if unknown:
            raise ValueError(f"Unknown training source: {', '.join(sorted(unknown))}")
    output.mkdir(parents=True, exist_ok=True)
    if excluded:
        result["ablation"] = write_ablation(dataset, output, data, images["train"], set(excluded))
    (output / "diagnosis.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "class_support.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "class", "images", "instances", "short_side_under_32px"])
        writer.writeheader()
        for split, stats in splits.items():
            for name, counts in stats["classes"].items():
                writer.writerow({"split": split, "class": name, **counts})
    print(f"Diagnosis: {output / 'diagnosis.json'}")
    print("Missing training classes:", ", ".join(result["missing_train_classes"]) or "none")
    print("Missing validation classes:", ", ".join(result["missing_val_classes"]) or "none")
    if excluded:
        print("Fixed-validation ablation:", result["ablation"]["data_yaml"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data/training")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--metadata-only", action="store_true",
                        help="Use saved image support; no label/image content reads or box counts")
    parser.add_argument("--exclude-training-source", action="append", default=[])
    args = parser.parse_args()
    if args.imgsz < 1:
        parser.error("imgsz must be positive")
    try:
        diagnose(args.dataset, args.output, args.run, args.imgsz,
                 args.exclude_training_source, args.metadata_only)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
