# YOLO model outputs

`scripts/train.py` puts everything about the **trained model** here:

- `weights/yolo11n.pt`: downloaded pretrained starting weights (local; Git-ignored).
- `runs/yolo11n-10x/weights/best.pt`: best trained checkpoint (local; Git-ignored).
- `runs/yolo11n-10x/results.csv`: training metrics and plots.
- `runs/evaluation_val*/metrics.json`: per-class validation report.
- `runs/evaluation_test*/metrics.json`: independent NHS test report, once collected.

No trained checkpoint is committed to GitHub. The repository contains training code and 3,096 original images; actual model fitting must run on your Mac or another machine.
