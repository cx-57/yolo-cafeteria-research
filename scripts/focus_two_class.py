"""Build and train a plate/bottle detector from the signed-off 900-image focus dataset.

The original three-class dataset and all existing checkpoints are never modified.
Wrapper boxes are removed; wrapper-only photographs become reviewed negatives.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil

import yaml

from focus_three_class import verify as verify_three
from prepare import PROJECT_ROOT
from v2 import digest

SOURCE = PROJECT_ROOT / "data/focus-three-class-improvement/all-reviewed-stack"
DATA = PROJECT_ROOT / "data/focus-two-class-900"
MODEL = PROJECT_ROOT / "model/focus-two-class-900"
NAMES = ["plate", "bottle"]


def two_class_rows(rows):
    """Three-class IDs: plate=0, wrapper=1, bottle=2; output: plate=0, bottle=1."""
    result = []
    for row in rows:
        parts = row.split()
        if len(parts) != 5 or parts[0] not in ("0", "1", "2"):
            raise ValueError("Unexpected source annotation: " + row)
        if parts[0] != "1":
            result.append(("0" if parts[0] == "0" else "1") + " " + " ".join(parts[1:]))
    return result


def summarize(records):
    counts = {split: dict(images=0, negatives=0, objects=dict.fromkeys(NAMES, 0),
                          class_images=dict.fromkeys(NAMES, 0))
              for split in ("train", "val")}
    for record in records:
        stats = counts[record["split"]]
        stats["images"] += 1
        stats["negatives"] += not record["rows"]
        classes = [int(row.split()[0]) for row in record["rows"]]
        for index, name in enumerate(NAMES):
            stats["objects"][name] += classes.count(index)
            stats["class_images"][name] += index in classes
    if any(not all(v["objects"].values()) for v in counts.values()):
        raise ValueError("Plate and bottle must both appear in training and validation")
    return counts


def original_records(source):
    verified = verify_three(source, allow_relocated=True)
    return verified, {(r["split"], r["name"]): r for r in verified["records"]}


def verify(data):
    data = data.resolve()
    manifest = json.loads((data / "manifest.json").read_text())
    source = Path(manifest["source"]).resolve()
    original, indexed = original_records(source)
    if original["fingerprint"] != manifest["source_fingerprint"]:
        raise ValueError("Original signed-off dataset changed; rebuild in a new directory")
    config = yaml.safe_load((data / "data.yaml").read_text())
    if config != dict(path=str(data), train="images/train", val="images/val", names=NAMES):
        raise ValueError("Two-class dataset config is invalid")
    expected_records = []
    for (split, name), record in indexed.items():
        expected_records.append(dict(split=split, name=name, rows=two_class_rows(record["rows"]),
                                     sha=record["sha"]))
    expected_records.sort(key=lambda item: (item["split"], item["name"]))
    if manifest["records"] != expected_records or manifest["counts"] != summarize(expected_records):
        raise ValueError("Two-class derivation manifest mismatch")
    for split in ("train", "val"):
        subset = [r for r in expected_records if r["split"] == split]
        expected_images = {r["name"] for r in subset}
        expected_labels = {str(Path(name).with_suffix(".txt")) for name in expected_images}
        images = data / "images" / split
        labels = data / "labels" / split
        if ({p.name for p in images.iterdir() if p.is_file()} != expected_images or
                {p.name for p in labels.iterdir() if p.is_file()} != expected_labels):
            raise ValueError("Missing or unexpected two-class files")
        for record in subset:
            image = images / record["name"]
            label = labels / Path(record["name"]).with_suffix(".txt")
            if digest(image) != record["sha"] or label.read_text().splitlines() != record["rows"]:
                raise ValueError("Altered image or labels: " + record["name"])
    return manifest


def build(source, data):
    source, data = source.resolve(), data.resolve()
    if data.exists() or data.is_symlink() or data == source or source in data.parents:
        raise ValueError("Choose a NEW dataset directory; source data is never overwritten")
    original, indexed = original_records(source)
    records = sorted((dict(split=split, name=name, rows=two_class_rows(r["rows"]), sha=r["sha"])
                      for (split, name), r in indexed.items()),
                     key=lambda item: (item["split"], item["name"]))
    counts = summarize(records)
    for split in ("train", "val"):
        (data / "images" / split).mkdir(parents=True)
        (data / "labels" / split).mkdir(parents=True)
    for r in records:
        image = source / "images" / r["split"] / r["name"]
        shutil.copy2(image, data / "images" / r["split"] / r["name"])
        (data / "labels" / r["split"] / Path(r["name"]).with_suffix(".txt")).write_text(
            "\n".join(r["rows"]) + ("\n" if r["rows"] else ""))
    (data / "data.yaml").write_text(yaml.safe_dump(
        dict(path=str(data), train="images/train", val="images/val", names=NAMES),
        sort_keys=False))
    (data / "manifest.json").write_text(json.dumps(dict(
        source=str(source), source_fingerprint=original["fingerprint"],
        policy="plate=0, bottle=1; wrapper boxes dropped; all reviewed photographs retained",
        records=records, counts=counts), indent=2) + "\n")
    verify(data)
    print("Built two-class dataset:", counts)


def add_minimum_ap_callback(model, output, validation_objects, target):
    best_score = -1.0

    def select(trainer):
        nonlocal best_score
        with Path(trainer.csv).open() as file:
            history = list(csv.DictReader(file))
        if not history or int(history[-1]["epoch"]) != trainer.epoch + 1:
            return
        metrics = trainer.validator.metrics
        indices = list(map(int, metrics.box.ap_class_index))
        if (set(indices) != {0, 1} or len(indices) != 2 or
                list(map(int, metrics.nt_per_class)) != validation_objects):
            raise ValueError("Validation must have both plate and bottle targets")
        scores = {NAMES[i]: float(metrics.box.ap50[j]) for j, i in enumerate(indices)}
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values()):
            raise ValueError("Invalid AP50")
        smallest = min(scores.values())
        if smallest > best_score:
            location = output / "train/weights/best_min_ap50.pt"
            location.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(trainer.last, location)
            best_score = smallest
            (output / "minimum_class_selection.json").write_text(json.dumps(dict(
                epoch=trainer.epoch + 1, per_class_AP50=scores, minimum_AP50=smallest,
                weights=str(location), weights_sha256=digest(location)), indent=2) + "\n")
        if smallest >= target:
            trainer.stop = True

    model.add_callback("on_fit_epoch_end", select)


def train(data, output, epochs, device, target):
    manifest = verify(data)
    if output.exists() or output.is_symlink():
        raise ValueError("Choose a new run directory; existing checkpoints are preserved")
    from ultralytics import YOLO
    pretrained = PROJECT_ROOT / "model/weights/yolo11n.pt"
    weights = str(pretrained) if pretrained.is_file() else "yolo11n.pt"
    output.mkdir(parents=True)
    (output / "experiment.json").write_text(json.dumps(dict(
        classes=NAMES, source_dataset_fingerprint=manifest["source_fingerprint"],
        epochs=epochs, pretrained=weights, target_min_AP50=target, device=device), indent=2) + "\n")
    model = YOLO(weights)  # Fresh COCO weights: do not reuse incompatible three-class heads.
    add_minimum_ap_callback(model, output,
                            [manifest["counts"]["val"]["objects"][n] for n in NAMES], target)
    model.train(data=str(data / "data.yaml"), epochs=epochs, batch=4, imgsz=640,
                workers=0, seed=42, patience=0, device=device, project=str(output),
                name="train", exist_ok=False, save=True)
    if not (output / "train/weights/best_min_ap50.pt").exists():
        raise ValueError("No valid two-class evaluation checkpoint was selected")


def evaluate(data, output, checkpoint, device, name):
    manifest = verify(data)
    from ultralytics import YOLO
    if not checkpoint.is_file():
        raise ValueError("Checkpoint not found: " + str(checkpoint))
    if (output / name).exists():
        raise ValueError("Evaluation output already exists")
    model = YOLO(str(checkpoint))
    classes = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
    if classes != NAMES:
        raise ValueError("Checkpoint does not have exactly plate and bottle classes")
    metrics = model.val(data=str(data / "data.yaml"), split="val", device=device,
                        batch=4, imgsz=640, workers=0, project=str(output), name=name,
                        plots=True)
    box = metrics.box
    indices = list(map(int, box.ap_class_index))
    if (set(indices) != {0, 1} or len(indices) != 2 or
            list(map(int, metrics.nt_per_class)) !=
            [manifest["counts"]["val"]["objects"][n] for n in NAMES]):
        raise ValueError("Validation support changed")
    vectors = {key: list(getattr(box, key)) for key in ("ap50", "ap", "p", "r")}
    if any(len(values) != len(indices) for values in vectors.values()):
        raise ValueError("Validation AP rows do not align with supported classes")
    macros = dict(ap50=float(box.map50), ap=float(box.map), p=float(box.mp), r=float(box.mr))
    values = list(macros.values()) + [float(v) for row in vectors.values() for v in row]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("Validation contains invalid or nonfinite metrics")
    for key, value in macros.items():
        if not math.isclose(sum(float(v) for v in vectors[key]) / len(indices), value,
                            abs_tol=0.001):
            raise ValueError("Validation macro disagrees with per-class " + key)
    verify(data)
    scores = {}
    for j, i in enumerate(indices):
        p, r = float(vectors["p"][j]), float(vectors["r"][j])
        scores[NAMES[i]] = dict(AP50=float(vectors["ap50"][j]), AP50_95=float(vectors["ap"][j]),
                               precision=p, recall=r, F1=2*p*r/(p+r) if p+r else 0.0,
                               objects=manifest["counts"]["val"]["objects"][NAMES[i]])
    result = dict(per_class=scores, mAP50=float(box.map50),
                  mAP50_95=float(box.map), weights=str(checkpoint),
                  weights_sha256=digest(checkpoint), imgsz=640, batch=4,
                  test_time_augmentation=False, independent_test=False,
                  validation_images=manifest["counts"]["val"]["images"],
                  source_dataset_fingerprint=manifest["source_fingerprint"],
                  note="Development validation only; not an independent NHS test")
    (output / name / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify", "train", "evaluate"))
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--target", type=float, default=0.90)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--eval-name", default="evaluation")
    args = parser.parse_args()
    try:
        if args.epochs < 1 or not 0 < args.target <= 1:
            raise ValueError("epochs and AP50 target must be positive and valid")
        if args.command == "build":
            build(args.source, args.data)
        elif args.command == "verify":
            print("Verified:", verify(args.data)["counts"])
        elif args.command == "train":
            train(args.data.resolve(), args.model.resolve(), args.epochs, args.device, args.target)
        else:
            weights = args.weights or (args.model / "train/weights/best_min_ap50.pt")
            evaluate(args.data.resolve(), args.model.resolve(), weights, args.device, args.eval_name)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
