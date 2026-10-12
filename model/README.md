# YOLO model outputs

`scripts/train.py` puts everything about the **trained model** here:

- `weights/yolo11n.pt`: downloaded pretrained starting weights (local; Git-ignored).
- `runs/yolo11n-1x/weights/best.pt`: best trained checkpoint (local; Git-ignored).
- `runs/yolo11n-1x/results.csv`: training metrics and plots.
- `runs/evaluation_val*/metrics.json`: per-class validation report.
- `runs/evaluation_test*/metrics.json`: independent NHS test report, once collected.

Evaluation reports per-class precision/recall/F1/AP, target support and missing
validation classes. It checks checkpoint class IDs and records validation size.
See the [plateau investigation](../data/audits/model_plateau_2026-10-11/README.md)
for measured baseline evidence and controlled fine-tuning commands. Source
ablation manifests and grouped splits are generated locally; existing
checkpoints and original annotations are preserved.

No trained checkpoint is committed to GitHub. The repository contains training code and 3,096 original images; actual model fitting must run on your Mac or another machine.
