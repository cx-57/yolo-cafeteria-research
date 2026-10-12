"""CPU-only Ultralytics integration smoke checks on tiny real repo photos.

Not a performance benchmark. Keeps all test artifacts outside tracked data.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from PIL import Image
import torch
from ultralytics import YOLO
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train

torch.set_num_threads(2)


def build_sample(folder):
    # Train and val use distinct originals and exercise both historical and
    # current-ID sources. Only the temporary training copies are written.
    selection = {
        "train": {
            "school_lunch": 2, "tetra_pak": 2, "plate": 2,
            "utensil": 2, "food_american": 2, "condiments": 1,
            "waste_detection_v3": 1,
        },
        "val": {"school_lunch": 2, "tetra_pak": 2, "macworx": 2},
    }
    used = set()
    label_total = 0
    for split, sources in selection.items():
        for source, needed in sources.items():
            originals = ROOT / "data" / "processed" / source / "images"
            count = 0
            for original in sorted(originals.iterdir()):
                if original.suffix.lower() not in train.IMAGE_SUFFIXES:
                    continue
                if original.resolve() in used:
                    continue
                old_label = ROOT / "data" / "processed" / source / "labels" / (original.stem + ".txt")
                boxes, _ = train.read_labels(old_label, source, train.class_names())
                with Image.open(original) as img:
                    img.verify()
                name = source + "__" + original.stem
                out_image = folder / "images" / split / (name + original.suffix)
                out_label = folder / "labels" / split / (name + ".txt")
                out_image.parent.mkdir(parents=True, exist_ok=True)
                out_label.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, out_image)
                out_label.write_text("\n".join(boxes) + "\n")
                for row in boxes:
                    parts = row.split()
                    cls_id = int(parts[0])
                    assert 0 <= cls_id < len(train.class_names()), row
                    assert len(parts) == 5, row
                label_total += len(boxes)
                used.add(original.resolve())
                count += 1
                if count == needed:
                    break
            assert count == needed, (split, source, count, needed)
    file = folder / "data.yaml"
    file.write_text(yaml.safe_dump({
        "path": str(folder.resolve()), "train": "images/train",
        "val": "images/val", "names": train.class_names(),
    }, sort_keys=False))
    assert label_total > 0
    assert len(used) == 18
    train.check_data(file)
    print("PASS real-photo import and ID remapping: 18 distinct photos", flush=True)
    return file


def run_training(data, name, model, manual=False):
    cmd = [
        sys.executable, str(ROOT / "scripts/train.py"), str(data),
        "--model", str(model), "--name", name, "--epochs", "1",
        "--batch", "2", "--imgsz", "128", "--device", "cpu",
        "--workers", "0", "--mosaic", "0", "--close-mosaic", "0",
    ]
    if manual:
        cmd += ["--optimizer", "AdamW", "--lr0", "0.0001", "--lrf", "0.1", "--cos-lr"]
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    weights = ROOT / "model" / "runs" / name / "weights" / "best.pt"
    assert weights.is_file() and weights.stat().st_size > 0
    report = ROOT / "model" / "runs" / "evaluation_val" / "metrics.json"
    # The automatic evaluator increments output folder for the second fit.
    if manual:
        options = sorted((ROOT / "model" / "runs").glob("evaluation_val*/metrics.json"),
                         key=lambda p: p.stat().st_mtime)
        assert options, "Missing per-class evaluation JSON"
        report = options[-1]
    assert report.is_file(), "Missing automatic validation metrics: " + str(report)
    metrics = json.loads(report.read_text())
    assert metrics["coverage"]["total_class_count"] == len(train.class_names())
    assert metrics["coverage"]["supported_class_count"] >= 1
    assert metrics["coverage"]["unsupported_class_count"] >= 1
    assert sum(x["support"] for x in metrics["per_class"].values()) > 0
    assert metrics["imgsz"] == 128, metrics
    print("PASS 1-epoch YOLO training and evaluation:", name, flush=True)
    return weights


def smoke_inference(weights, data):
    model = YOLO(str(weights))
    assert list(model.names.values()) == train.class_names()
    image = next((data.parent / "images/val").glob("*"))
    predictions = model.predict(str(image), imgsz=128, device="cpu", verbose=False)
    assert len(predictions) == 1
    p = predictions[0]
    assert p.orig_shape[0] > 0 and p.orig_shape[1] > 0
    if len(p.boxes):
        assert torch.isfinite(p.boxes.xyxy).all()
        assert torch.all((p.boxes.cls >= 0) & (p.boxes.cls < len(train.class_names())))
    print("PASS saved checkpoint inference:", len(p.boxes), "detections", flush=True)


def main():
    with tempfile.TemporaryDirectory(prefix="yolo-smoke-") as temporary:
        data = build_sample(Path(temporary))
        original = run_training(data, "ci-smoke-from-yaml", "yolo11n.yaml")
        smoke_inference(original, data)
        tuned = run_training(data, "ci-smoke-adamw-finetune", original, manual=True)
        smoke_inference(tuned, data)
        print("ALL YOLO INTEGRATION SMOKES PASSED", flush=True)


if __name__ == "__main__":
    main()
