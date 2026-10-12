"""V2 dataset and controlled training workflow. Existing V1 files are read-only.

Usage: python3 scripts/v2.py build --help
       python3 scripts/v2.py train --help
       python3 scripts/v2.py report --help
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import yaml

from experiment import check_output, label_path, prepare_experiment, validate_rows
from prepare import IMAGE_SUFFIXES, PROJECT_ROOT
from train import check_names, class_names


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def source_group(filename):
    """Group re-exports of a source photo, regardless of Roboflow hash/split."""
    stem = Path(filename).stem.lower().split(".rf.")[0]
    if "__" in stem:
        prefix, original = stem.split("__", 1)
        original = re.sub(r"^(train|valid|val|test)__", "", original)
        return prefix + "__" + original
    return stem


def load_dataset(data_file):
    data_file = Path(data_file).resolve()
    spec = yaml.safe_load(data_file.read_text())
    check_names(spec["names"], str(data_file))
    root = Path(spec.get("path") or data_file.parent)
    if not root.is_absolute():
        root = data_file.parent / root
    result = {}
    for split in ("train", "val", "test"):
        if split not in spec:
            continue
        folder = Path(spec[split])
        if not folder.is_absolute():
            folder = root / folder
        if not folder.is_dir():
            raise ValueError(f"Missing {split} image directory: {folder}")
        result[split] = sorted(
            path for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    if not result.get("train") or not result.get("val"):
        raise ValueError("Both train and val must contain images")
    return result


def check_nhs_manifest(manifest, source_files):
    """Require human-verified, exhaustive labels and capture-group separation."""
    if manifest is None:
        return []
    manifest = Path(manifest).resolve()
    with manifest.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"image", "label", "split", "session", "reviewer", "reviewed", "complete"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"NHS CSV requires columns: {', '.join(sorted(required))}")
        records = list(reader)
    seen_paths, sessions, images = set(), {}, []
    source_hashes = {digest(file) for files in source_files.values() for file in files}
    for line, record in enumerate(records, 2):
        if record["split"] not in ("train", "val", "test"):
            raise ValueError(f"NHS row {line}: split must be train, val, or test")
        if record["reviewed"].strip().lower() != "yes" or record["complete"].strip().lower() != "yes":
            raise ValueError(f"NHS row {line}: reviewed and complete must both equal yes")
        session, reviewer = record["session"].strip(), record["reviewer"].strip()
        if not session or not reviewer:
            raise ValueError(f"NHS row {line}: session and reviewer are required")
        if session in sessions and sessions[session] != record["split"]:
            raise ValueError(f"NHS capture session crosses splits: {session}")
        sessions[session] = record["split"]
        image = (manifest.parent / record["image"]).resolve()
        label = (manifest.parent / record["label"]).resolve()
        if image.suffix.lower() not in IMAGE_SUFFIXES or not image.is_file() or not label.is_file():
            raise ValueError(f"NHS row {line}: image or label is missing/unsupported")
        if image in seen_paths:
            raise ValueError(f"NHS image listed multiple times: {image}")
        seen_paths.add(image)
        validate_rows([v for v in label.read_text().splitlines() if v.strip()],
                      label, class_names())
        image_hash = digest(image)
        if image_hash in source_hashes:
            raise ValueError(f"NHS image duplicates a legacy source image: {image}")
        images.append({"image": image, "label": label, "split": record["split"],
                       "session": session, "reviewer": reviewer, "sha256": image_hash})
    if len({r["sha256"] for r in images}) != len(images):
        raise ValueError("NHS manifest contains duplicate image contents")
    return images


def audit_splits(sources, nhs):
    """Catch exact and source-name leakage without altering any source."""
    groups, sha_groups, classes = {}, {}, {}
    names = class_names()
    for split, photos in sources.items():
        classes[split] = {name: 0 for name in names}
        for photo in photos:
            key = source_group(photo.name)
            if key in groups and groups[key] != split:
                raise ValueError(f"Legacy original/Roboflow group crosses splits: {key}")
            groups[key] = split
            image_sha = digest(photo)
            if image_sha in sha_groups and sha_groups[image_sha] != split:
                raise ValueError(f"Identical source photo crosses splits: {photo}")
            sha_groups[image_sha] = split
            label = label_path(photo)
            if not label.is_file():
                raise ValueError(f"Missing label: {label}")
            for row in validate_rows([r for r in label.read_text().splitlines()
                                      if r.strip()], label, names):
                classes[split][names[int(row.split()[0])]] += 1
    for photo in nhs:
        split = photo["split"]
        classes.setdefault(split, {name: 0 for name in names})
        if photo["sha256"] in sha_groups and sha_groups[photo["sha256"]] != split:
            raise ValueError(f"Identical NHS photo crosses splits: {photo['image']}")
        sha_groups[photo["sha256"]] = split
        for row in photo["label"].read_text().splitlines():
            if row.strip():
                classes[split][names[int(row.split()[0])]] += 1
    missing = {split: [name for name, count in classes.get(split, {}).items() if count == 0]
               if split in classes else names[:] for split in ("train", "val", "test")}
    return {"instances": classes, "missing_classes": missing, "image_counts": {
        split: len(sources.get(split, [])) + sum(r["split"] == split for r in nhs)
        for split in ("train", "val", "test")}}


def dataset_fingerprint(data_file):
    """Image and label content fingerprint. Detect post-build dataset changes."""
    files = load_dataset(data_file)
    sha = hashlib.sha256()
    for split, photos in sorted(files.items()):
        for image in photos:
            label = label_path(image)
            if not label.is_file():
                raise ValueError(f"Missing label: {label}")
            for entry in (image, label):
                sha.update(f"{split}/{entry.name}\0".encode())
                sha.update(digest(entry).encode())
    return sha.hexdigest()



def validate_overrides(file, sources):
    """Approve FULL replacement label contents for specific V2 images only."""
    if file is None:
        return []
    raw = json.loads(Path(file).read_text())
    if not isinstance(raw, list):
        raise ValueError("Reviewed overrides must be a JSON list")
    available = {photo.stem: (split, photo) for split, images in sources.items()
                 for photo in images}
    result, seen = [], set()
    for number, record in enumerate(raw, 1):
        if not isinstance(record, dict):
            raise ValueError(f"Override {number}: expected object")
        source, image = record.get("source"), record.get("image")
        if not isinstance(source, str) or not isinstance(image, str):
            raise ValueError(f"Override {number}: source and image required")
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", source) or Path(image).name != image:
            raise ValueError(f"Override {number}: invalid source/image path")
        stem = source + "__" + Path(image).stem
        if stem not in available or stem in seen:
            raise ValueError(f"Override {number}: missing or duplicate source image {stem}")
        seen.add(stem)
        split, photo = available[stem]
        label = label_path(photo)
        if record.get("complete") != "yes" or not record.get("reviewer") or not record.get("reason"):
            raise ValueError(f"Override {number}: requires complete=yes, reviewer and reason")
        if digest(label) != record.get("expected_label_sha256"):
            raise ValueError(f"Override {number}: source label changed since manual review")
        if not isinstance(record.get("rows"), list):
            raise ValueError(f"Override {number}: rows must contain the COMPLETE approved labels")
        approved = validate_rows(record["rows"], f"Override {number}", class_names())
        result.append({"stem": stem, "split": split, "rows": approved,
                       "reviewer": record["reviewer"], "reason": record["reason"],
                       "original_sha256": digest(label)})
    return result


def build(args):
    source = Path(args.source).resolve()
    source_yaml = source if source.suffix in (".yaml", ".yml") else source / "data.yaml"
    sources = load_dataset(source_yaml)
    dest = Path(args.output).resolve()
    check_output(dest, [source_yaml.parent, *[photo.parent for photos in sources.values()
                                            for photo in photos]])
    nhs = check_nhs_manifest(args.nhs_manifest, sources)
    overrides = validate_overrides(args.reviewed_overrides, sources)
    # In strict V2, ALL public-source V1 photos are train-only.
    # Validation and test must consist solely of independently reviewed NHS captures.
    audited_sources = sources if args.provisional else {
        "train": sources["train"] + sources["val"]
    }
    evidence = audit_splits(audited_sources, nhs)
    for record in overrides:
        split = record["split"] if args.provisional else "train"
        prior = label_path(next(photo for photo in sources[record["split"]]
                                if photo.stem == record["stem"]))
        for row in [r for r in prior.read_text().splitlines() if r.strip()]:
            name = class_names()[int(row.split()[0])]
            evidence["instances"][split][name] -= 1
        for row in record["rows"]:
            name = class_names()[int(row.split()[0])]
            evidence["instances"][split][name] += 1
    evidence["missing_classes"] = {
        split: [name for name in class_names()
                if evidence["instances"].get(split, {}).get(name, 0) == 0]
        for split in ("train", "val", "test")
    }
    if not args.provisional:
        if any(evidence["missing_classes"][split] for split in ("train", "val", "test")):
            raise ValueError("V2 class coverage is incomplete. Supply reviewed NHS photos "
                             "for all 11 classes, or use --provisional for a NONFINAL build. "
                             + json.dumps(evidence["missing_classes"]))
        if not any(p["split"] == "test" for p in nhs):
            raise ValueError("Final V2 requires an independently collected NHS test split")
    # The existing experiment builder validates additive boxes and never edits V1.
    prepare_experiment(source_yaml, dest, label_additions=args.label_additions)
    if not args.provisional:
        # V1 public-source validation is NOT a domain-matched V2 evaluation set.
        # Reassign only V2 COPIES of legacy validation images to training.
        for image in list((dest / "images/val").iterdir()):
            target = dest / "images/train" / image.name
            target_label = dest / "labels/train" / (image.stem + ".txt")
            old_label = dest / "labels/val" / (image.stem + ".txt")
            if target.exists() or target_label.exists():
                raise ValueError(f"V2 copy collision: {image.name}")
            image.rename(target)
            old_label.rename(target_label)
    for record in overrides:
        output_split = record["split"] if args.provisional else "train"
        output_label = dest / "labels" / output_split / (record["stem"] + ".txt")
        if not output_label.is_file():
            raise ValueError(f"Missing V2 label copy: {output_label}")
        output_label.write_text("\n".join(record["rows"]) +
                                ("\n" if record["rows"] else ""))
    spec_file = dest / "data.yaml"
    spec = yaml.safe_load(spec_file.read_text())
    nhs_summary = []
    for index, record in enumerate(nhs, 1):
        split = record["split"]
        image = record["image"]
        name = f"nhs__{index:05d}_{image.name}"
        target_image = dest / "images" / split / name
        target_label = dest / "labels" / split / (Path(name).stem + ".txt")
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        # Use independent bytes for labels; source labels and photos never edited.
        shutil.copy2(image, target_image)
        shutil.copy2(record["label"], target_label)
        nhs_summary.append({"image": name, "split": split, "session": record["session"],
                            "reviewer": record["reviewer"],
                            "source_sha256": record["sha256"],
                            "label_sha256": digest(record["label"])})
    if any(r["split"] == "test" for r in nhs):
        spec["test"] = str((dest / "images/test").resolve())
    spec_file.write_text(yaml.safe_dump(spec, sort_keys=False))
    final = audit_splits(load_dataset(spec_file), [])
    # Source group enforcement applies to the new final dataset too.
    if not args.provisional and any(final["missing_classes"][k] for k in ("train", "val", "test")):
        raise ValueError("V2 dataset is still incomplete after copying")
    summary = {"provisional": args.provisional, "source_yaml": str(source_yaml),
               "nhs_manifest": str(Path(args.nhs_manifest).resolve()) if args.nhs_manifest else None,
               "nhs_photos": nhs_summary, "source_audit": evidence,
               "final_audit": final, "dataset_sha256": dataset_fingerprint(spec_file),
               "annotation_policy": "docs/REDESIGN_V2.md (2026-10-12)",
               "strict_nhs_only_evaluation": not args.provisional,
               "label_additions": str(args.label_additions) if args.label_additions else None,
               "reviewed_overrides": overrides}
    (dest / "v2_build.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("V2 dataset:", spec_file)
    print("Provisional:", args.provisional)
    print("Missing classes:", final["missing_classes"])
    return spec_file


TRIALS = {
    "control-n": {"model": "yolo11n.pt"},
    "weighted-n": {"model": "yolo11n.pt", "cls_pw": "0.25"},
    "control-s": {"model": "yolo11s.pt"},
}


def train(args):
    data_file = Path(args.data).resolve()
    summary_file = data_file.parent / "v2_build.json"
    if not summary_file.is_file():
        raise ValueError("Expected dataset prepared by scripts/v2.py build")
    summary = json.loads(summary_file.read_text())
    if summary["provisional"] and not args.allow_provisional:
        raise ValueError("Dataset marked provisional. Use --allow-provisional only for a "
                         "clearly labeled engineering experiment, not for an accuracy claim.")
    if dataset_fingerprint(data_file) != summary["dataset_sha256"]:
        raise ValueError("V2 dataset content changed since build; rebuild and re-audit")
    trial = TRIALS[args.trial]
    tag = args.tag or ("e" + str(args.epochs))
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise ValueError("--tag may contain only letters, digits, dashes and underscores")
    run_name = "v2-" + args.trial + "-" + tag
    evaluation_name = run_name + "-evaluation"
    run_dir = PROJECT_ROOT / "model/runs" / run_name
    evaluation_dir = PROJECT_ROOT / "model/runs" / evaluation_name
    if not args.dry_run and (run_dir.exists() or evaluation_dir.exists()):
        raise ValueError(f"Run already exists. Choose another --tag: {run_name}")
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts/train.py"), str(data_file),
           "--model", trial["model"], "--name", run_name,
           "--eval-name", evaluation_name,
           "--epochs", str(args.epochs), "--batch", str(args.batch),
           "--imgsz", "640", "--seed", "42", "--device", args.device,
           "--optimizer", "AdamW", "--lr0", "0.0007", "--lrf", "0.1",
           "--cos-lr", "--mosaic", "0.3", "--close-mosaic", "10",
           "--workers", "0"]
    if "cls_pw" in trial:
        cmd.extend(["--cls-pw", trial["cls_pw"]])
    print("Training:", " ".join(cmd), flush=True)
    if not args.dry_run:
        subprocess.run(cmd, check=True)
        if not run_dir.is_dir():
            raise ValueError(f"Completed training run directory missing: {run_dir}")
        (run_dir / "v2_trial.json").write_text(json.dumps({
            "trial": args.trial, "run_name": run_name,
            "evaluation_name": evaluation_name,
            "dataset_sha256": summary["dataset_sha256"],
            "data": str(data_file), "epochs": args.epochs,
            "batch": args.batch, "seed": 42, "imgsz": 640,
            "provisional": summary["provisional"],
            "command": cmd,
        }, indent=2) + "\n")


def report(args):
    """Compare native Ultralytics metrics only when class support matches."""
    entries = []
    tag = args.tag or ("e" + str(args.epochs))
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise ValueError("Invalid report --tag")
    signatures = set()
    for trial in args.trials:
        run_name = "v2-" + trial + "-" + tag
        recipe = PROJECT_ROOT / "model/runs" / run_name / "v2_trial.json"
        if not recipe.is_file():
            raise ValueError(f"Missing training run manifest: {recipe}")
        trial_info = json.loads(recipe.read_text())
        signatures.add(trial_info["dataset_sha256"])
        path = PROJECT_ROOT / "model/runs" / (run_name + "-evaluation") / "metrics.json"
        if not path.is_file():
            raise ValueError(f"Missing evaluated checkpoint: {path}")
        metrics = json.loads(path.read_text())
        coverage = metrics.get("coverage", {})
        if coverage.get("unsupported_class_count", 11) and not args.allow_incomplete:
            raise ValueError(f"{trial}: incomplete class evaluation; cannot rank all 11 classes")
        entries.append({"trial": trial, "map50": metrics["map50"],
                        "map50_95": metrics["map50_95"], "precision": metrics["precision"],
                        "recall": metrics["recall"], "per_class": metrics["per_class"],
                        "coverage": coverage, "data": metrics.get("data")})
    if len({e["data"] for e in entries}) != 1 or len(signatures) != 1:
        raise ValueError("Trials used different dataset paths or contents")
    for row in sorted(entries, key=lambda r: r["map50_95"], reverse=True):
        print(f"{row['trial']:12s} mAP50-95={row['map50_95']:.3f}, "
              f"mAP50={row['map50']:.3f}, P={row['precision']:.3f}, "
              f"R={row['recall']:.3f}")
    print("Provisional public-source metrics are NOT independent NHS accuracy.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    building = commands.add_parser("build", help="Prepare audited V2 dataset copies")
    building.add_argument("--source", default=str(PROJECT_ROOT / "data/training"))
    building.add_argument("--output", default=str(PROJECT_ROOT / "data/training-v2"))
    building.add_argument("--nhs-manifest", type=Path)
    building.add_argument("--label-additions", type=Path)
    building.add_argument("--reviewed-overrides", type=Path)
    building.add_argument("--provisional", action="store_true")
    training = commands.add_parser("train", help="Controlled YOLO11n/s trial")
    training.add_argument("--data", default=str(PROJECT_ROOT / "data/training-v2/data.yaml"))
    training.add_argument("--trial", choices=list(TRIALS), required=True)
    training.add_argument("--epochs", type=int, default=80)
    training.add_argument("--batch", type=int, default=4)
    training.add_argument("--device", default="mps")
    training.add_argument("--tag", help="Unique trial output tag (default e<epochs>)")
    training.add_argument("--dry-run", action="store_true")
    training.add_argument("--allow-provisional", action="store_true")
    comparison = commands.add_parser("report", help="Compare completed trials")
    comparison.add_argument("--trials", nargs="+", choices=list(TRIALS),
                            default=list(TRIALS))
    comparison.add_argument("--allow-incomplete", action="store_true")
    comparison.add_argument("--epochs", type=int, default=80)
    comparison.add_argument("--tag", help="Trial output tag; default e<epochs>")
    args = parser.parse_args()
    try:
        if args.command == "build":
            build(args)
        elif args.command == "train":
            if args.epochs < 1 or args.batch < 1:
                raise ValueError("--epochs and --batch must be positive")
            train(args)
        else:
            report(args)
    except (OSError, ValueError, yaml.YAMLError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
