import argparse

import yaml
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("experiment")
parser.add_argument("--data", required=True)
args = parser.parse_args()

with open(args.experiment, "r") as file:
    config = yaml.safe_load(file)

model = YOLO(config["model"])

model.train(
    data=args.data,
    imgsz=config.get("imgsz", 640),
    epochs=config.get("epochs", 100),
    project="runs",
    name=config["name"],
)
