"""Minimal Ultralytics training entry point."""

from __future__ import annotations
import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--data", required=True, help="Prepared Ultralytics data.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.experiment.read_text())
    model = YOLO(cfg["model"])
    model.train(
        data=args.data,
        imgsz=cfg.get("imgsz", 640),
        epochs=cfg.get("epochs", 100),
        project="runs",
        name=cfg["name"],
    )

if __name__ == "__main__":
    main()
