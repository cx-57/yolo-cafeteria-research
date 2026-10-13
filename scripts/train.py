"""Prepare the 80/20 dataset, train YOLO11, and evaluate the model."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil

from PIL import Image
import yaml

from prepare import PROJECT_ROOT, IMAGE_SUFFIXES, LEGACY_SOURCES, current_label, remap_label_line

DATA = PROJECT_ROOT / "data"
OUTPUT = DATA / "training"
MODEL = PROJECT_ROOT / "model"
TRAIN_ONLY = {
    "condiments", "tartanhacks", "waste_detection_v3",
    "dry_wet_waste", "school_lunch_asian", "food_american",
}
SOURCES = LEGACY_SOURCES | TRAIN_ONLY


def class_names():
    classes = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())["classes"]
    return [classes[i] for i in range(len(classes))]


def group_for_split(source, filename):
    """Group multiple Roboflow exports of the same original photo."""
    name = re.sub(r"^(train|valid|val|test)__", "", filename).split(".rf.")[0]
    if source == "tartanhacks":
        match = re.search(r"IMG_\d+", name)
        if match:
            return match.group()
    return name


def read_labels(path, source, names):
    """Validate boxes, correct tiny export-rounding errors, preserve source files."""
    valid = []
    corrected = 0
    names_by_id = dict(enumerate(names))
    ids_by_name = {name: i for i, name in enumerate(names)}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = current_label(line, source).split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{number}: expected 5 numbers")
        x, y, w, h = map(float, parts[1:])
        left, right, top, bottom = x-w/2, x+w/2, y-h/2, y+h/2
        excess = max(0, -left, right-1, -top, bottom-1)
        if 0 < excess <= 0.0001 and w > 0 and h > 0:
            left, right = max(0, left), min(1, right)
            top, bottom = max(0, top), min(1, bottom)
            parts[1:] = [f"{(left+right)/2:.9f}", f"{(top+bottom)/2:.9f}",
                         f"{right-left:.9f}", f"{bottom-top:.9f}"]
            corrected += 1
        row = remap_label_line(" ".join(parts), names_by_id,
                               ids_by_name, path, number)
        if row is None:
            raise ValueError(f"{path}:{number}: unrecognized class")
        valid.append(row)
    if not valid:
        raise ValueError(f"Missing object annotations: {path}")
    return valid, corrected


def collect_images(names):
    """Read originals; never use data/raw or unreviewed candidates for training."""
    groups, duplicates = defaultdict(list), defaultdict(set)
    sources, corrections = Counter(), Counter()
    for source in sorted(SOURCES):
        images = DATA / "processed" / source / "images"
        labels = DATA / "processed" / source / "labels"
        if not images.is_dir():
            continue
        for image in sorted(images.iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = labels / (image.stem + ".txt")
            rows, fixed = read_labels(label, source, names)
            with Image.open(image) as opened:
                rgb = opened.convert("RGB")
                pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            group = source + ":" + group_for_split(source, image.stem)
            groups[group].append((source, image, rows))
            duplicates[pixels].add(group)
            sources[source] += 1
            corrections[source] += fixed
        for label in labels.glob("*.txt"):
            if not any((images / (label.stem + ext)).exists() for ext in IMAGE_SUFFIXES):
                raise ValueError(f"Label without image: {label}")
    if not groups:
        raise ValueError("No processed photos found")

    # Exact pixel duplicates must stay in the same split, even across sources.
    parent = {key: key for key in groups}

    def root(key):
        while key != parent[key]:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for matching in duplicates.values():
        first = min(matching)
        for key in matching:
            parent[root(key)] = root(first)
    combined = defaultdict(list)
    for key, photos in groups.items():
        combined[root(key)].extend(photos)
    return combined, sources, corrections


def grouped_candidates(groups, audit_file):
    """Keep audited near-duplicate candidates together without changing originals."""
    report = {"path": None, "candidate_pairs": 0, "matched_pairs": 0,
              "out_of_corpus_pairs": 0, "merged_groups": 0}
    if audit_file is None:
        return groups, report
    audit_file = Path(audit_file)
    report["path"] = str(audit_file.resolve())
    index = {f"{source}/images/{image.name}": group
             for group, photos in groups.items() for source, image, _ in photos}
    parent = {group: group for group in groups}

    def root(group):
        while group != parent[group]:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    try:
        with audit_file.open(newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            fields = reader.fieldnames or []
            if not {"image_a", "image_b"} <= set(fields) or len(fields) != len(set(fields)):
                raise ValueError("expected image_a and image_b columns")
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"row {reader.line_num}: wrong number of columns")
                paths = [row[field].strip() for field in ("image_a", "image_b")]
                for path in paths:
                    parts = path.split("/")
                    if (len(parts) != 3 or parts[1] != "images"
                            or any(part in {"", ".", ".."} for part in parts)
                            or any(char in path for char in "\\\n\r\0")
                            or Path(path).suffix.lower() not in IMAGE_SUFFIXES):
                        raise ValueError(f"row {reader.line_num}: expected source/images/filename")
                report["candidate_pairs"] += 1
                if any(path not in index for path in paths):
                    report["out_of_corpus_pairs"] += 1
                    continue
                report["matched_pairs"] += 1
                first, second = (root(index[path]) for path in paths)
                if first != second:
                    parent[max(first, second)] = min(first, second)
                    report["merged_groups"] += 1
    except (OSError, csv.Error, ValueError) as error:
        raise ValueError(f"Invalid group audit {audit_file}: {error}") from error
    combined = defaultdict(list)
    for group, photos in groups.items():
        combined[root(group)].extend(photos)
    return combined, report


def check_output(output):
    """Only replace a recognizable generated dataset, never original data."""
    resolved = output.resolve()
    protected = ((DATA / "processed").resolve(), (DATA / "raw").resolve(), MODEL.resolve())
    if any(resolved == source or resolved in source.parents or source in resolved.parents
           for source in protected) or resolved == PROJECT_ROOT.resolve():
        raise ValueError(f"Unsafe output path overlaps original data or model files: {output}")
    if output.is_symlink():
        raise ValueError(f"Unsafe output symlink: {output}")
    if not output.exists():
        return
    try:
        metadata = yaml.safe_load((output / "data.yaml").read_text())
        summary = json.loads((output / "split_summary.json").read_text())
        if (Path(metadata["path"]).resolve() != resolved
                or "original_images" not in summary
                or set(path.name for path in output.iterdir())
                - {"images", "labels", "data.yaml", "split_summary.json"}):
            raise ValueError("directory is not a generated training dataset")
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        raise ValueError(f"Refusing to replace unrelated output directory: {output}") from error


def build_dataset(augment_factor=1, seed=42, group_audit=None, output=None):
    if not 1 <= augment_factor <= 20:
        raise ValueError("Augmentation factor must be 1–20")
    output = Path(output) if output is not None else OUTPUT
    check_output(output)
    names = class_names()
    groups, source_counts, corrections = collect_images(names)
    resolved_output = output.resolve()
    if any(resolved_output in source_path.resolve().parents
           for photos in groups.values() for source, image, _ in photos
           for source_path in (image, DATA / "processed" / source / "labels" / (image.stem + ".txt"))):
        raise ValueError(f"Unsafe output contains a training source image or label: {output}")
    # Conservative leakage prevention is the default when the tracked near-
    # duplicate audit exists. A candidate pair is grouped, never auto-deleted.
    if group_audit is None:
        known_candidates = DATA / "audits/day1/near_duplicates.csv"
        if known_candidates.is_file():
            group_audit = known_candidates
    groups, grouping = grouped_candidates(groups, group_audit)
    total = sum(source_counts.values())
    target = round(total * 0.20)

    # Pick ~20% of ALL originals for validation, but never split a photo group
    # or put known correlated training-only sources into validation.
    eligible = sorted(
        (key for key, photos in groups.items()
         if not any(source in TRAIN_ONLY for source, _, _ in photos)),
        key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).digest(),
    )
    validation = set()
    selected = 0
    for key in eligible:
        if selected + len(groups[key]) <= target:
            validation.add(key)
            selected += len(groups[key])

    if not validation:
        raise ValueError("No independent validation photos available")
    if output.exists():
        shutil.rmtree(output)  # Validated generated copies only.

    counts = Counter()
    class_counts = {"train": Counter(), "val": Counter()}
    for group, photos in sorted(groups.items()):
        split = "val" if group in validation else "train"
        for source, image, rows in photos:
            stem = source + "__" + image.stem
            image_dir = output / "images" / split
            label_dir = output / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            output_image = image_dir / (stem + image.suffix)
            output_label = label_dir / (stem + ".txt")
            if output_image.exists() or output_label.exists():
                raise ValueError(f"Duplicate output name: {stem}")
            try:
                os.link(image, output_image)
            except OSError:
                shutil.copy2(image, output_image)
            output_label.write_text("\n".join(rows) + "\n")
            counts[split] += 1
            class_counts[split].update(set(int(row.split()[0]) for row in rows))

            # Hardlinks avoid storing 10 physical copies. Ultralytics applies
            # random, bbox-aware augmentations to each training view every epoch.
            if split == "train":
                for n in range(1, augment_factor):
                    suffix = f"__view{n:02d}"
                    os.link(output_image, image_dir / (stem + suffix + image.suffix))
                    os.link(output_label, label_dir / (stem + suffix + ".txt"))

    if counts["val"] != target:
        print(f"Grouped images prevent exact 80/20: {counts['val']} val, target {target}")
    data_yaml = output / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(output.resolve()), "train": "images/train",
        "val": "images/val", "names": names}, sort_keys=False))
    report = {
        "original_images": total, "train_originals": counts["train"],
        "val_originals": counts["val"], "test_originals": 0,
        "target_validation_originals": target,
        "train_views": counts["train"] * augment_factor,
        "augment_factor": augment_factor, "seed": seed,
        "output": str(output.resolve()), "group_audit": grouping,
        "source_images": dict(source_counts),
        "minor_boundary_rounding_clips_in_generated_labels": dict(corrections),
        "train_class_images_original": {name: class_counts["train"][i] for i, name in enumerate(names)},
        "val_class_images_original": {name: class_counts["val"][i] for i, name in enumerate(names)},
        "missing_train_classes": [name for i, name in enumerate(names) if not class_counts["train"][i]],
        "missing_val_classes": [name for i, name in enumerate(names) if not class_counts["val"][i]],
    }
    (output / "split_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{total} originals: {counts['train']} train, {counts['val']} val, "
          f"{report['train_views']} training views ({augment_factor}x)")
    print("No test photos included. Collect independent NHS test photos.")
    if grouping["path"]:
        print(f"Conservatively grouped {grouping['matched_pairs']} audited candidate pairs; "
              f"skipped {grouping['out_of_corpus_pairs']} out-of-corpus pairs. "
              "Candidate pairs do not prove duplicates.")
    if report["missing_train_classes"]:
        print("Missing training classes:", ", ".join(report["missing_train_classes"]))
    if report["missing_val_classes"]:
        print("Missing validation classes:", ", ".join(report["missing_val_classes"]))
    return data_yaml


def check_names(names, location):
    expected = class_names()
    if isinstance(names, dict):
        if len(names) != len(expected):
            raise ValueError(f"{location}: wrong class names or IDs")
        names = [names.get(i, names.get(str(i))) for i in range(len(expected))]
    if names != expected:
        raise ValueError(f"{location}: wrong class names or IDs")


def check_data(data_file):
    data = yaml.safe_load(Path(data_file).read_text())
    check_names(data["names"], "data.yaml")
    return data


def evaluate(weights, data_file, split="val", device=None, imgsz=None, batch=8,
             workers=8, name=None):
    data = check_data(data_file)
    if split == "test" and not data.get("test"):
        raise ValueError("Add independent test: images/path to your data.yaml first")
    model = load_model(str(weights))
    check_names(model.names, "checkpoint")
    options = dict(data=str(data_file), split=split, project=str(MODEL / "runs"),
                   name=name or "evaluation_" + split, plots=True,
                   batch=batch, workers=workers)
    if imgsz is not None:
        options["imgsz"] = imgsz
    if device:
        options["device"] = device

    # The validator normalizes image size and may reduce the batch to the
    # dataset size. Record those settings, including checkpoint defaults.
    effective = {}
    validator_support = {}

    def record_settings(validator):
        dataloader = getattr(validator, "dataloader", None)
        effective.update(imgsz=int(validator.args.imgsz), batch=int(
            getattr(dataloader, "batch_size", None) or validator.args.batch))
        for key in ("nt_per_class", "nt_per_image"):
            value = getattr(validator, key, None)
            if value is not None:
                validator_support[key] = value

    model.add_callback("on_val_end", record_settings)
    metrics = model.val(**options)
    # Refuse partial, misaligned, or nonfinite validator output. In particular,
    # zip() alone can silently drop a supported class and inflate a headline mAP.
    names = class_names()
    support = getattr(metrics, "nt_per_class", None)
    images = getattr(metrics, "nt_per_image", None)
    if support is None:
        support = validator_support.get("nt_per_class")
    if images is None:
        images = validator_support.get("nt_per_image")
    if support is None or images is None or len(support) != len(names) or len(images) != len(names):
        raise ValueError("Validation class support does not match the 11-class ontology")
    if any(int(n) != n or n < 0 for n in [*support, *images]):
        raise ValueError("Validation contains invalid target counts")
    indices = [int(i) for i in metrics.box.ap_class_index]
    vectors = {key: list(getattr(metrics.box, key))
               for key in ("ap50", "ap", "p", "r", "f1")}
    if (len(indices) != len(set(indices))
            or any(len(v) != len(indices) for v in vectors.values())
            or set(indices) != {i for i, n in enumerate(support) if n > 0}):
        raise ValueError("Validation AP rows do not exactly match supported classes")
    scalars = [float(getattr(metrics.box, key))
               for key in ("mp", "mr", "map50", "map")]
    values = scalars + [float(n) for vector in vectors.values() for n in vector]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("Validation contains invalid or nonfinite metrics")
    if not indices:
        raise ValueError("Validation has no ground-truth targets; AP is undefined")
    for field, overall in (("ap50", metrics.box.map50), ("ap", metrics.box.map),
                           ("p", metrics.box.mp), ("r", metrics.box.mr)):
        mean = sum(float(v) for v in vectors[field]) / len(indices)
        if not math.isclose(mean, float(overall), abs_tol=0.001):
            raise ValueError(f"Validation macro {field} disagrees with per-class results")
    measured = {
        i: {"ap50": float(vectors["ap50"][j]), "map50_95": float(vectors["ap"][j]),
            "precision": float(vectors["p"][j]), "recall": float(vectors["r"][j]),
            "f1": float(vectors["f1"][j])}
        for j, i in enumerate(indices)
    }
    # A single detected item can produce a nearly perfect AP. Requiring a
    # minimum count does not invent accuracy; it gates public performance claims.
    minimum_support = 30
    per_class = {}
    for i, class_name in enumerate(class_names()):
        row = dict.fromkeys(("ap50", "map50_95", "precision", "recall", "f1"))
        count = int(support[i])
        status = "unsupported"
        if count:
            status = "supported" if i in measured else "metrics_unavailable"
            row.update(measured.get(i, {}))
        row.update(support=count, images=int(images[i]), status=status,
                   reliability=("absent" if count == 0 else
                                "insufficient_support" if count < minimum_support else
                                "measurable_public_validation"))
        per_class[class_name] = row
    supported = [class_name for class_name, row in per_class.items() if row["support"]]
    unsupported = [class_name for class_name, row in per_class.items() if not row["support"]]
    limited = [class_name for class_name, row in per_class.items()
               if 0 < row["support"] < minimum_support]
    result = {
        "split": split, "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr), "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class": per_class,
        "coverage": {"total_class_count": len(per_class),
                     "supported_class_count": len(supported),
                     "unsupported_class_count": len(unsupported),
                     "all_classes_evaluable": not unsupported,
                     "minimum_instances_per_class_for_reliable_reporting": minimum_support,
                     "insufficient_support_classes": limited,
                     "all_classes_adequately_supported": not unsupported and not limited,
                     "supported_classes": supported, "unsupported_classes": unsupported},
        "interpretation": ("Only classes with ground-truth targets enter macro AP. "
                           "A test split name does not prove independence."
                           if split == "test" else
                           "Provisional validation only; not independent NHS test performance. "
                           "Classes without targets have undefined AP."),
        "weights": str(Path(model.ckpt_path if hasattr(model, "ckpt_path")
                            else weights).resolve()),
        "data": str(Path(data_file).resolve()),
        **effective,
    }
    location = Path(metrics.save_dir) / "metrics.json"
    location.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Results:", location)
    if unsupported:
        print("INCOMPLETE CLASS COVERAGE: AP undefined for", ", ".join(unsupported))
    if limited:
        print("INSUFFICIENT VALIDATION SUPPORT (<30 objects):", ", ".join(limited))
    return result


def load_model(name):
    from ultralytics import YOLO
    name = str(name)
    if not re.fullmatch(r"yolo11[^/\\]*\.pt", name):
        return YOLO(name)
    weights = MODEL / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    target = weights / name
    if target.exists():
        return YOLO(str(target))
    # Ultralytics downloads pretrained YOLO11 into the current directory.
    # Download into model/weights/ rather than cluttering the project root.
    original = Path.cwd()
    try:
        os.chdir(weights)
        return YOLO(str(target.resolve()))
    finally:
        os.chdir(original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", type=Path, help="Prepared data.yaml")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--output", type=Path, help="Directory for a newly built dataset")
    parser.add_argument("--group-audit", type=Path,
                        help="Audited near_duplicates.csv; conservatively group candidate pairs")
    parser.add_argument("--augment-factor", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int,
                        help="Image size: train defaults to 640; evaluate uses checkpoint size")
    parser.add_argument("--device", help="mps on Apple Silicon; 0 on NVIDIA")
    parser.add_argument("--name", help="Training run name; evaluation name with --evaluate")
    parser.add_argument("--eval-name", help="Name for the evaluation output directory")
    parser.add_argument("--optimizer", choices=("auto", "AdamW", "Adam", "SGD", "RMSProp"),
                        default="auto")
    parser.add_argument("--lr0", type=float, help="Initial learning rate; requires a manual optimizer")
    parser.add_argument("--lrf", type=float, help="Final learning rate as a fraction of lr0 (0–1)")
    parser.add_argument("--cls-pw", type=float, default=None,
                        help="Optional inverse-frequency class weighting power (0–1)")
    parser.add_argument("--freeze", type=int, default=None,
                        help="Optional number of initial layers to freeze during fine-tuning")
    parser.add_argument("--cos-lr", action="store_true", help="Use cosine learning rate decay")
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--mosaic", type=float, default=0.3)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--warmup-epochs", type=float, default=3.0,
                        help="Warmup duration; use 0 for a short low-rate fine-tune")
    parser.add_argument("--coverage-policy", type=Path,
                        help="Experimental source-coverage loss policy; requires --mosaic 0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", type=Path, metavar="LAST_PT")
    parser.add_argument("--evaluate", type=Path, metavar="BEST_PT")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()
    if (args.output is not None or args.group_audit is not None) and (
            args.data is not None or args.evaluate or args.resume):
        parser.error("Each of --output and --group-audit only applies when building a dataset")
    if min(args.epochs, args.batch, args.imgsz if args.imgsz is not None else 640) < 1:
        parser.error("epochs, batch and imgsz must be positive")
    if min(args.patience, args.close_mosaic, args.workers) < 0:
        parser.error("patience, close-mosaic and workers must be nonnegative")
    if not math.isfinite(args.warmup_epochs) or args.warmup_epochs < 0:
        parser.error("warmup-epochs must be finite and nonnegative")
    for option in ("mosaic", "lrf", "cls_pw"):
        value = getattr(args, option)
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            parser.error(f"--{option} must be finite and between 0 and 1")
    if args.freeze is not None and args.freeze < 0:
        parser.error("--freeze must be a nonnegative integer")
    if args.lr0 is not None:
        if not math.isfinite(args.lr0) or args.lr0 <= 0:
            parser.error("--lr0 must be finite and positive")
        if args.optimizer == "auto":
            parser.error("--lr0 requires an explicit --optimizer because auto ignores lr0")
    if args.coverage_policy and (args.resume or args.evaluate):
        parser.error("--coverage-policy applies to fresh training only, not resume/evaluation")
    if args.coverage_policy and args.mosaic != 0:
        parser.error("--coverage-policy requires --mosaic 0 (no mixed-image augmentation)")
    if args.resume:
        from ultralytics import YOLO
        YOLO(str(args.resume)).train(resume=True)
        return
    if args.evaluate:
        evaluate(args.evaluate, args.data or OUTPUT / "data.yaml", args.split, args.device,
                 imgsz=args.imgsz, batch=args.batch, workers=args.workers,
                 name=args.eval_name or args.name)
        return
    if args.split == "test":
        parser.error("--split test requires --evaluate")
    imgsz = args.imgsz if args.imgsz is not None else 640
    data_file = args.data or build_dataset(args.augment_factor, args.seed,
                                         group_audit=args.group_audit, output=args.output)
    check_data(data_file)
    if args.build_only:
        return

    import ultralytics
    if ultralytics.__version__ != "8.4.175":
        print("WARNING: Ultralytics", ultralytics.__version__,
              "differs from repository-tested 8.4.175. "
              "Do not treat this as a controlled comparison.")
    model = load_model(args.model)
    options = dict(
        data=str(data_file), project=str(MODEL / "runs"), name=args.name or "yolo11n-1x",
        epochs=args.epochs, batch=args.batch, imgsz=imgsz, seed=args.seed,
        optimizer=args.optimizer, cos_lr=args.cos_lr, workers=args.workers,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience, plots=True, save=True,
        degrees=8, translate=0.1, scale=0.2, fliplr=0.5,
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.3, mosaic=args.mosaic,
        close_mosaic=args.close_mosaic, mixup=0.0, cutmix=0.0, copy_paste=0.0)
    if args.cls_pw is not None:
        options["cls_pw"] = args.cls_pw
    if args.freeze is not None:
        options["freeze"] = args.freeze
    if args.lr0 is not None:
        options["lr0"] = args.lr0
    if args.lrf is not None:
        options["lrf"] = args.lrf
    if args.device:
        options["device"] = args.device
    if args.coverage_policy:
        from partial_labels import make_trainer
        options["trainer"] = make_trainer(args.coverage_policy)
    model.train(**options)
    evaluate(Path(model.trainer.save_dir) / "weights/best.pt", data_file,
             device=args.device, imgsz=imgsz, batch=args.batch, workers=args.workers,
             name=args.eval_name)


if __name__ == "__main__":
    main()
