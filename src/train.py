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

    # Mild train-time augmentation for realistic cafeteria variation.
    degrees=10,
    translate=0.10,
    scale=0.25,
    fliplr=0.5,
    flipud=0.0,
    hsv_h=0.01,
    hsv_s=0.4,
    hsv_v=0.3,
    mosaic=0.3,
    mixup=0.0,
)
