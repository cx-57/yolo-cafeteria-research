"""Actual YOLO11n training, evaluation and prediction smoke on corrected V2 photos.

One epoch is an integration check, not a model-quality benchmark.
"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile

from PIL import Image
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train import class_names

torch.set_num_threads(2)


def make_cohort(v2_data, target):
    spec = yaml.safe_load(Path(v2_data).read_text())
    names = class_names()
    requested = {
        "train": {"food": 2, "plate": 2, "utensil": 2,
                  "hinged_container": 1, "drink_carton": 1,
                  "wrapper": 2, "bottle": 2, "napkin": 1,
                  "condiment_packet": 1, "bowl": 1},
        "val": {"food": 2, "plate": 2, "wrapper": 1, "bottle": 1}
    }
    totals = {}
    for split, wanted in requested.items():
        source_images = Path(spec[split])
        source_labels = Path(str(source_images).replace("/images/", "/labels/"))
        image_index = {p.stem: p for p in source_images.iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")}
        pairs = []
        for label in sorted(source_labels.glob("*.txt")):
            if label.stem not in image_index:
                raise ValueError("Unpaired source V2 label: " + str(label))
            ids = {int(row.split()[0]) for row in label.read_text().splitlines()
                   if row.strip()}
            pairs.append((image_index[label.stem], label, ids))
        selected = {}
        used = set()
        for name, count in wanted.items():
            cls = names.index(name)
            options = [(im, lb) for im, lb, ids in pairs if cls in ids and im not in used]
            if len(options) < count:
                raise ValueError("Insufficient V2 smoke examples for " + name)
            for image, label in options[:count]:
                used.add(image)
                selected[image] = label
        for image, label in selected.items():
            with Image.open(image) as opened:
                opened.verify()
            images_dir = target / "images" / split
            labels_dir = target / "labels" / split
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, images_dir / image.name)
            shutil.copy2(label, labels_dir / label.name)
        totals[split] = len(selected)
    yaml_path = target / "data.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "path": str(target.resolve()), "train": "images/train",
        "val": "images/val", "names": names
    }, sort_keys=False))
    print("V2 checked cohort:", totals, flush=True)
    return yaml_path


def smoke(data_file, weighted):
    name = "ci-v2-weighted-n" if weighted else "ci-v2-control-n"
    cmd = [sys.executable, str(ROOT / "scripts/train.py"), str(data_file),
           "--model", "yolo11n.pt", "--name", name, "--eval-name", name + "-evaluation",
           "--epochs", "1", "--imgsz", "128", "--batch", "2",
           "--device", "cpu", "--workers", "0", "--seed", "42",
           "--optimizer", "AdamW", "--lr0", "0.0007", "--lrf", "0.1",
           "--cos-lr", "--mosaic", "0", "--close-mosaic", "0",
           "--warmup-epochs", "0", "--patience", "0"]
    if weighted:
        cmd += ["--cls-pw", "0.25"]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    run = ROOT / "model/runs" / name
    checkpoint = run / "weights/best.pt"
    metrics_path = ROOT / "model/runs" / (name + "-evaluation") / "metrics.json"
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise AssertionError("No trained YOLO checkpoint: " + str(checkpoint))
    metrics = json.loads(metrics_path.read_text())
    assert metrics["coverage"]["supported_class_count"] >= 1
    assert metrics["imgsz"] == 128
    assert 0 <= metrics["map50"] <= 1
    model = YOLO(str(checkpoint))
    image = next((data_file.parent / "images/val").iterdir())
    result = model.predict(str(image), imgsz=128, device="cpu", verbose=False)
    assert len(result) == 1
    if len(result[0].boxes):
        assert torch.isfinite(result[0].boxes.xyxy).all()
    print("PASS pretrained V2 1-epoch fit, validation, checkpoint reload, inference:", name,
          "diagnostic mAP50=", metrics["map50"], flush=True)


def main():
    v2_data = ROOT / "data/training-ci-v2-reviewed/data.yaml"
    if not v2_data.is_file():
        raise ValueError("First build V2 via --use-codex-review")
    with tempfile.TemporaryDirectory(prefix="v2-real-smoke-") as tmp:
        data = make_cohort(v2_data, Path(tmp))
        smoke(data, weighted=False)
        smoke(data, weighted=True)
    print("ALL V2 END-TO-END CPU TRAINING SMOKES PASSED", flush=True)


if __name__ == "__main__":
    main()
