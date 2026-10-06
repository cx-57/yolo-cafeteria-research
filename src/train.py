import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("data", help="Path to YOLO data.yaml")
parser.add_argument("--model", default="yolo26n.pt")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--imgsz", type=int, default=640)
args = parser.parse_args()

model = YOLO(args.model)

model.train(
    data=args.data,
    epochs=args.epochs,
    imgsz=args.imgsz,
    project="runs",
)
