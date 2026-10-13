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


def check_nhs_manifest(manifest, source_files, allow_unverified=False):
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
        reviewed = record["reviewed"].strip().lower()
        complete = record["complete"].strip().lower()
        required_status = "no" if allow_unverified else "yes"
        if reviewed != required_status or complete != required_status:
            raise ValueError(f"NHS row {line}: reviewed and complete must both equal "
                             f"{required_status}")
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
                       "session": session, "reviewer": reviewer, "sha256": image_hash,
                       "source_url": record.get("source_url"), "license": record.get("license"),
                       "attribution": record.get("attribution")})
    if len({r["sha256"] for r in images}) != len(images):
        raise ValueError("NHS manifest contains duplicate image contents")
    return images


def repair_legacy_split_groups(sources):
    """Plan transitive train/val repairs for V2 copies, without touching V1."""
    all_photos = [(split, image) for split, items in sources.items() for image in items]
    parent = list(range(len(all_photos)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    groups, hashes = {}, {}
    for i, (_split, image) in enumerate(all_photos):
        group, sha = source_group(image.name), digest(image)
        if group in groups:
            union(i, groups[group])
        else:
            groups[group] = i
        if sha in hashes:
            union(i, hashes[sha])
        else:
            hashes[sha] = i

    components = {}
    for i, entry in enumerate(all_photos):
        components.setdefault(find(i), []).append(entry)

    moved, repairs = set(), []
    for entries in components.values():
        splits = {split for split, _image in entries}
        if len(splits) < 2:
            continue
        if "test" in splits:
            raise ValueError("Legacy test images conflict with other splits: " +
                             ", ".join(str(p) for _, p in entries))
        if splits == {"train", "val"}:
            affected = sorted((p for split, p in entries if split == "val"))
            moved.update(affected)
            repairs.append({
                "reason": "Shared original name or exact photo hash",
                "component_images": [str(p) for _, p in sorted(entries, key=lambda r: str(r[1]))],
                "moved_validation_images": [str(p) for p in affected]
            })

    if moved and len(moved) == len(sources["val"]):
        raise ValueError("All validation images overlap training; collect a clean validation set")
    planned = {}
    for split, items in sources.items():
        if split == "train":
            planned[split] = sorted(items + list(moved))
        elif split == "val":
            planned[split] = [p for p in items if p not in moved]
        else:
            planned[split] = items
    return planned, moved, repairs


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
    reviewed_audit = bool(getattr(args, "use_codex_review", False))
    if reviewed_audit and (args.label_additions or args.reviewed_overrides):
        raise ValueError("--use-codex-review already supplies both approved files; "
                         "do not combine it with explicit overrides/additions")
    audit_root = PROJECT_ROOT / "data/audits/v2_review_2026-10-12"
    label_additions = (audit_root / "lvis_food_approved.json" if reviewed_audit
                       else args.label_additions)
    reviewed_overrides = (audit_root / "reviewed_overrides.json" if reviewed_audit
                          else args.reviewed_overrides)
    public_manifest = getattr(args, "reviewed_manifest", None)
    nhs = check_nhs_manifest(args.nhs_manifest, sources)
    for record in nhs:
        record["domain"] = "nhs"
    public = check_nhs_manifest(public_manifest, sources)
    for record in public:
        record["domain"] = "reviewed_public"
    nhs.extend(public)
    candidate_manifest = getattr(args, "candidate_manifest", None)
    if candidate_manifest and not args.provisional:
        raise ValueError("Unverified candidate photos are allowed only with --provisional")
    candidates = check_nhs_manifest(candidate_manifest, sources,
                                    allow_unverified=True)
    for record in candidates:
        if record["split"] != "train":
            raise ValueError("Unverified candidate photos must stay in training")
        record["domain"] = "unverified_candidate"
    nhs.extend(candidates)
    if len({r["sha256"] for r in nhs}) != len(nhs):
        raise ValueError("Additional manifests contain duplicate image contents")
    sessions = {}
    for record in nhs:
        if record["session"] in sessions and sessions[record["session"]] != record["split"]:
            raise ValueError("Additional capture session crosses splits: " + record["session"])
        sessions[record["session"]] = record["split"]
    overrides = validate_overrides(reviewed_overrides, sources)
    # A provisional split can contain duplicated Roboflow originals from V1.
    # Repair only V2 copies. Strict mode makes ALL legacy images training-only.
    if args.provisional:
        audited_sources, moved_validation, split_repairs = repair_legacy_split_groups(sources)
    else:
        audited_sources = {"train": sources["train"] + sources["val"]}
        moved_validation = set(sources["val"])
        split_repairs = [{"reason": "Legacy sources are training-only in reviewed V2; "
                         "this keeps every legacy capture and near-duplicate group in one split.",
                         "moved_validation_images": [str(p) for p in sources["val"]]}]
    evidence = audit_splits(audited_sources, nhs)
    for record in overrides:
        split = ("train" if any(p.stem == record["stem"] for p in moved_validation)
                 else record["split"])
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
    # A test set is desirable for final claims, but not required for training
    # or development comparison on fully reviewed NHS validation photos.
    has_test = any(p["split"] == "test" for p in nhs)
    required_splits = ("train", "val", "test") if has_test else ("train", "val")
    if not args.provisional:
        if any(evidence["missing_classes"][split] for split in required_splits):
            raise ValueError("V2 class coverage is incomplete. Supply reviewed additional photos "
                             "for all classes in train and validation "
                             "(and test if present), or use --provisional. "
                             + json.dumps(evidence["missing_classes"]))
    # The existing experiment builder validates additive boxes and never edits V1.
    prepare_experiment(source_yaml, dest, label_additions=label_additions)
    # Reassign ONLY V2 image and label copies; the existing V1 split is untouched.
    for image in sorted(moved_validation):
        original = dest / "images/val" / image.name
        updated = dest / "images/train" / image.name
        original_label = dest / "labels/val" / (image.stem + ".txt")
        updated_label = dest / "labels/train" / (image.stem + ".txt")
        if not original.is_file() or not original_label.is_file():
            raise ValueError(f"V2 image/label copy not found: {image.name}")
        if updated.exists() or updated_label.exists():
            raise ValueError(f"Duplicate V2 destination: {image.name}")
        original.rename(updated)
        original_label.rename(updated_label)
    for record in overrides:
        output_split = ("train" if any(p.stem == record["stem"] for p in moved_validation)
                        else record["split"])
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
        namespace = {"nhs": "nhs", "reviewed_public": "reviewed",
                     "unverified_candidate": "candidate"}[record["domain"]]
        name = f"{namespace}__{index:05d}_{image.name}"
        target_image = dest / "images" / split / name
        target_label = dest / "labels" / split / (Path(name).stem + ".txt")
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        # Use independent bytes for labels; source labels and photos never edited.
        shutil.copy2(image, target_image)
        shutil.copy2(record["label"], target_label)
        nhs_summary.append({"image": name, "split": split, "session": record["session"],
                            "reviewer": record["reviewer"],
                            "domain": record["domain"], "source_url": record["source_url"],
                            "license": record["license"], "attribution": record["attribution"],
                            "source_sha256": record["sha256"],
                            "label_sha256": digest(record["label"])})
    if any(r["split"] == "test" for r in nhs):
        spec["test"] = str((dest / "images/test").resolve())
    spec_file.write_text(yaml.safe_dump(spec, sort_keys=False))
    final = audit_splits(load_dataset(spec_file), [])
    # Source group enforcement applies to the new final dataset too.
    if not args.provisional and any(final["missing_classes"][k] for k in required_splits):
        raise ValueError("V2 dataset is still incomplete after copying")
    summary = {"provisional": args.provisional, "source_yaml": str(source_yaml),
               "nhs_manifest": str(Path(args.nhs_manifest).resolve()) if args.nhs_manifest else None,
               "reviewed_manifest": str(Path(public_manifest).resolve()) if public_manifest else None,
               "nhs_photos": nhs_summary,
               "unverified_candidate_images": len(candidates),
               "source_audit": evidence,
               "legacy_split_repairs": split_repairs,
               "legacy_validation_images_moved": len(moved_validation),
               "final_audit": final, "dataset_sha256": dataset_fingerprint(spec_file),
               "annotation_policy": "docs/REDESIGN_V2.md (2026-10-12)",
               "strict_nhs_only_evaluation": not args.provisional and not public_manifest,
               "evaluation_domain": ("provisional_legacy" if args.provisional else
                                     "reviewed_public" if public_manifest else "nhs"),
               "independent_test_available": has_test,
               "evaluation_note": ("Independent NHS test split available."
                                   if has_test else
                                   "No independent test: compare on validation only; "
                                   "do not claim final unbiased generalization."),
               "label_additions": str(label_additions) if label_additions else None,
               "reviewed_audit_applied": reviewed_audit,
               "reviewed_overrides": overrides}
    (dest / "v2_build.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("V2 dataset:", spec_file)
    print("Provisional:", args.provisional)
    if reviewed_audit:
        print(f"Codex-reviewed food additions and {len(overrides)} reviewed "
              "replacement annotation files applied to V2 copies")
    print("Independent test set:", "yes" if has_test else "no (validation only)")
    if split_repairs:
        print(f"Resolved {len(split_repairs)} leaking photo groups by moving "
              f"{len(moved_validation)} validation copies into V2 train")
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
           "--workers", "0", "--patience", "0"]
    if "cls_pw" in trial:
        cmd.extend(["--cls-pw", trial["cls_pw"]])
    print("Training:", " ".join(cmd), flush=True)
    if not args.dry_run:
        subprocess.run(cmd, check=True)
        if not run_dir.is_dir():
            raise ValueError(f"Completed training run directory missing: {run_dir}")
        with (run_dir / "results.csv").open(newline="") as stream:
            completed = list(csv.DictReader(stream))
        if len(completed) != args.epochs:
            raise ValueError(f"Incomplete trial: {len(completed)} of {args.epochs} epochs")
        (run_dir / "v2_trial.json").write_text(json.dumps({
            "trial": args.trial, "run_name": run_name,
            "evaluation_name": evaluation_name,
            "dataset_sha256": summary["dataset_sha256"],
            "data": str(data_file), "epochs": args.epochs,
            "batch": args.batch, "seed": 42, "imgsz": 640,
            "completed_epochs": len(completed), "device": args.device,
            "provisional": summary["provisional"],
            "command": cmd,
        }, indent=2) + "\n")
    return cmd


def check_trial_schedules(recipes):
    """Only model size and cls_pw may differ in the controlled comparison."""
    for field in ("epochs", "batch", "seed", "imgsz", "device", "dataset_sha256"):
        if len({recipe.get(field) for recipe in recipes}) != 1:
            raise ValueError("Trial schedule differs: " + field)
    schedules = []
    for recipe in recipes:
        if recipe.get("completed_epochs", recipe["epochs"]) != recipe["epochs"]:
            raise ValueError("Trial schedule was not completed")
        command = recipe.get("command", [])
        ignored = {"--model", "--name", "--eval-name", "--cls-pw"}
        schedule, i = [], 2
        while i < len(command):
            if command[i] in ignored:
                i += 2
                continue
            schedule.append(command[i])
            i += 1
        schedules.append(tuple(schedule))
    if len(set(schedules)) != 1:
        raise ValueError("Trial command schedule differs")


def report(args):
    """Compare native Ultralytics metrics only when class support matches."""
    entries = []
    tag = args.tag or ("e" + str(args.epochs))
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise ValueError("Invalid report --tag")
    signatures = set()
    recipes = []
    for trial in args.trials:
        run_name = "v2-" + trial + "-" + tag
        recipe = PROJECT_ROOT / "model/runs" / run_name / "v2_trial.json"
        if not recipe.is_file():
            raise ValueError(f"Missing training run manifest: {recipe}")
        trial_info = json.loads(recipe.read_text())
        recipes.append(trial_info)
        if trial_info.get("provisional") and not args.allow_incomplete:
            raise ValueError(f"{trial}: provisional dataset cannot be ranked as reviewed validation")
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
    check_trial_schedules(recipes)
    supports = [{name: row["support"] for name, row in e["per_class"].items()} for e in entries]
    if any(support != supports[0] for support in supports):
        raise ValueError("Trials evaluated different class support")
    for row in sorted(entries, key=lambda r: r["map50_95"], reverse=True):
        print(f"{row['trial']:12s} mAP50-95={row['map50_95']:.3f}, "
              f"mAP50={row['map50']:.3f}, P={row['precision']:.3f}, "
              f"R={row['recall']:.3f}")
    print("These metrics compare models on validation, not an independent final test.")



def make_review_queue(data_file, output, overrides_file=None, food_file=None):
    """List all images and explicitly distinguish partial from full visual sign-off."""
    dataset = load_dataset(data_file)
    review_root = PROJECT_ROOT / "data/audits/v2_review_2026-10-12"
    overrides_file = Path(overrides_file or review_root / "reviewed_overrides.json")
    food_file = Path(food_file or review_root / "lvis_food_approved.json")
    overrides = json.loads(overrides_file.read_text())
    food = json.loads(food_file.read_text())
    if not isinstance(overrides, list) or not isinstance(food, list):
        raise ValueError("Review files must contain lists")
    full_names = {r["source"] + "__" + Path(r["image"]).stem for r in overrides}
    # Manually inspected images with already-correct labels need no override.
    # Do not count thumbnail-only, ambiguous, or needs_followup decisions.
    review_csv = review_root / "image_review.csv"
    unchanged_hashes = {}
    if review_csv.is_file():
        with review_csv.open(newline="") as stream:
            for record in csv.DictReader(stream):
                if (record.get("review_scope") == "full_resolution"
                        and record.get("decision") == "checked_existing"):
                    source, filename = record.get("source", ""), record.get("image", "")
                    if not re.fullmatch(r"[a-zA-Z0-9_-]+", source) or Path(filename).name != filename:
                        raise ValueError("Invalid direct visual-review record: " + str(record))
                    fingerprint = record.get("label_sha256", "")
                    if not re.fullmatch(r"[a-f0-9]{64}", fingerprint):
                        raise ValueError("Checked-unchanged review requires original label SHA256")
                    stem = source + "__" + Path(filename).stem
                    if stem in unchanged_hashes and fingerprint != unchanged_hashes[stem]:
                        raise ValueError("Conflicting verified label hashes: " + stem)
                    unchanged_hashes[stem] = fingerprint
                    full_names.add(stem)
    food_names = {r["source"] + "__" + Path(r["image"]).stem for r in food}
    rows = []
    counts = {"full_image_verified": 0, "food_only_verified": 0, "not_verified": 0}
    names = class_names()
    for split, photos in dataset.items():
        for photo in photos:
            label = label_path(photo)
            if not label.is_file():
                raise ValueError("Missing review-queue label: " + str(label))
            lines = [line for line in label.read_text().splitlines() if line.strip()]
            ids = sorted({int(row.split()[0])
                          for row in validate_rows(lines, label, names)})
            if photo.stem in unchanged_hashes and digest(label) != unchanged_hashes[photo.stem]:
                raise ValueError("Visually verified unchanged label has changed; re-review: "
                                 + str(label))
            if photo.stem in full_names:
                status = "full_image_verified"
            elif photo.stem in food_names:
                status = "food_only_verified"
            else:
                status = "not_verified"
            counts[status] += 1
            rows.append({
                "split": split, "image": str(photo),
                "source_group": source_group(photo.name),
                "classes": "|".join(names[i] for i in ids),
                "instances": len(lines), "review_status": status,
                "requires_all_11_class_review": (
                    "no" if status == "full_image_verified" else "yes"),
                "label_sha256": digest(label)
            })
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        fields = ("split", "image", "source_group", "classes", "instances",
                  "review_status", "requires_all_11_class_review", "label_sha256")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("V2 image-level review queue:", output)
    print("Review counts:", counts, "total", len(rows))
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    building = commands.add_parser("build", help="Prepare audited V2 dataset copies")
    building.add_argument("--source", default=str(PROJECT_ROOT / "data/training"))
    building.add_argument("--output", default=str(PROJECT_ROOT / "data/training-v2"))
    building.add_argument("--nhs-manifest", type=Path)
    building.add_argument("--candidate-manifest", type=Path,
                          help="Explicit TRAIN-ONLY unverified image/label drafts; "
                               "requires --provisional and reviewed=no, complete=no")
    building.add_argument("--reviewed-manifest", type=Path,
                          help="Reviewed public/local photos with exhaustive labels and capture sessions; not NHS")
    building.add_argument("--label-additions", type=Path)
    building.add_argument("--reviewed-overrides", type=Path)
    building.add_argument("--use-codex-review", action="store_true",
                          help="Apply the 242 reviewed LVIS food overlays and approved "
                               "V2-only full-label replacements from 2026-10-12")
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
    queue = commands.add_parser("review-queue", help="Generate per-photo 11-class review register")
    queue.add_argument("--data", default=str(PROJECT_ROOT / "data/training/data.yaml"))
    queue.add_argument("--output", default=str(PROJECT_ROOT / "data/audits/v2_review_2026-10-12/review_queue_generated.csv"))
    queue.add_argument("--reviewed-overrides", type=Path)
    queue.add_argument("--food-reviews", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "build":
            build(args)
        elif args.command == "train":
            if args.epochs < 1 or args.batch < 1:
                raise ValueError("--epochs and --batch must be positive")
            train(args)
        elif args.command == "review-queue":
            make_review_queue(args.data, args.output, args.reviewed_overrides,
                              args.food_reviews)
        else:
            report(args)
    except (OSError, ValueError, yaml.YAMLError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
