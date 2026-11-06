# Unified YOLO V5 — interrupted 11-epoch checkpoint

**Status:** model checkpoint received and verified locally; **weights are not on GitHub yet**. The training notebook, validation diagnostics, and metadata are committed here. This is **not** the final frozen V1 specialist system and does not replace its weights or paper results.

## Uploaded checkpoint

- Source filename: `best(1).pt`
- Suggested GitHub Release filename: `YOLO_V5_11epochs_best.pt`
- Size: **94,869,987 bytes** (~90.5 MiB)
- SHA256: `62711f9deb1bb104eb23aabf5ae6fa83cba011c0aacee62877d04a77b827af27`
- Ultralytics version embedded: `8.4.203`
- Epoch recorded in checkpoint: `10` (zero-based; **11 completed epochs**, of 20 planned)
- Checkpoint timestamp embedded: `2026-11-06T04:12:58.402089+00:00`

**Metrics directly embedded in the checkpoint** (validation; not independently re-evaluated):

| Metric | Value |
| --- | ---: |
| Precision (B) | 0.69911 |
| Recall (B) | 0.62426 |
| mAP50 (B) | **0.65944** |
| mAP50–95 (B) | **0.41966** |

Model: `YOLO11s`, seven classes in order: `food, plate, utensil, hinged_container, drink_carton, wrapper, bottle`. Training used the separate approximately 80/10/10 V5 resplit experiment (2,261 source images total). Exact split allocation and full training logs should be retained alongside this model for independent replication.

The old 97-image V5 validation split included just four classes, so its scores **cannot be compared directly** to the new seven-class V5 evaluation. The V5 annotations and class completeness remain provisional. The result is **not** an independent, field-validated disposal accuracy figure.

## Files already on GitHub

- [Balanced V5 training / evaluation notebook](YOLO_V5_Balanced_80_10_10_20_Epochs.ipynb)
- [Validation findings and limitations](V5_VALIDATION_FINDINGS.md)
- [Checkpoint metadata JSON](V5_11Epoch_Checkpoint_Metadata.json)
- [Provisional data annotation audit](ANNOTATION_AUDIT_STATUS.md)

## How to publish the actual binary weights

The available GitHub connector can write UTF-8 repository files but **cannot directly transfer an attached 94.9-MB PyTorch binary**. The repo's `.gitignore` also intentionally excludes `*.pt`. **No model download URL is valid until the weight file is actually uploaded.**

Recommended: [create a GitHub Release](https://github.com/cx-57/yolo-cafeteria-research/releases/new) and attach the local `best(1).pt` renamed to `YOLO_V5_11epochs_best.pt`. GitHub Release assets support this file size. After uploading, calculate SHA256 of the downloaded asset and verify it matches the value above.

Alternatively, on a local machine with Git and Git LFS configured, publish in a separate nonfrozen experiment via LFS rather than ordinary Git history.

## Inference after downloading a verified checkpoint

```python
from ultralytics import YOLO
model = YOLO("YOLO_V5_11epochs_best.pt")
predictions = model.predict("photo.jpg", imgsz=800, conf=0.25)
for result in predictions:
    result.save(filename="v5_prediction.jpg")
```

Do not label the weights a fully trained 20-epoch checkpoint or suggest that this metadata commit uploaded the binary.
