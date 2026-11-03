# YOLO11n — 704-image grouped-split experiment

**Experimental model only.** This folder is independent of the four frozen V1 specialist detectors. No existing weights or inference code are changed.

## Dataset

| Source | Photos |
|---|---:|
| FLIC cafeteria trays | 104 |
| Canteen Food Waste | 175 |
| PUST processed cafeteria images | 425 |
| **Total** | **704** |

Group-safe split: **536 train, 97 validation, 71 test** (362 source/capture groups; 3,879 YOLO boxes).

| Class | Boxes |
|---|---:|
| food | 2,266 |
| plate | 1,130 |
| utensil | 376 |
| hinged_container | 0 |
| drink_carton | 0 |
| wrapper | 103 |
| bottle | 4 |

**Source archive:** `YOLO_V2_704_GroupSplit_Trial.zip` (136 MiB compressed approximately; exact SHA-256 below). The archive contains `yolo_v2_trial_704/` with all 704 original image bytes, 704 corresponding labels, `data.yaml`, a source/sha256/group manifest, and a verified training/evaluation script.

**Data not stored directly in GitHub yet.** GitHub's single-file limit is 100 MB and the complete ZIP is larger. Get the original ZIP from the YOLO Command Center chat attachment; do not substitute the public unannotated source datasets. If making this fully GitHub-hosted, publish it separately in GitHub Releases or Git LFS rather than committing one oversize archive. The files under this GitHub folder are experiment code, instructions, and metadata.

```
SHA-256 of YOLO_V2_704_GroupSplit_Trial.zip
16c9990b2cf45a14915ae3344307f4c1bf8dc9dbbd91a31d0b697313a8c21e27
```

## Run on Apple Silicon Mac

1. Download `YOLO_V2_704_GroupSplit_Trial.zip` from the conversation.
2. Unzip it. It creates the folder `yolo_v2_trial_704/`.
3. From Terminal run:

```bash
cd /path/to/yolo_v2_trial_704
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r /path/to/this/github/folder/requirements.txt
python3 train_yolo_v2_trial.py --verify-only
python3 train_yolo_v2_trial.py --device mps --epochs 35 --batch 4
```

For a self-contained download, you can instead install with `python3 -m pip install ultralytics==8.4.175` inside the extracted ZIP folder; the ZIP contains the same verified training script.

The experiment initializes **new `yolo11n.pt` pretrained weights** (automatically downloaded by Ultralytics) and saves new weights as `YOLO_V2_trial_results/train/weights/best.pt`. Final metrics are saved to `YOLO_V2_trial_results/test_summary.json`, including test mAP50, mAP50-95, precision, recall, and per-supported-class AP.

## Colab GPU

Use [`YOLO_V2_704_Colab_Train.ipynb`](YOLO_V2_704_Colab_Train.ipynb), select a GPU runtime, and upload the same ZIP. This does not overwrite the frozen specialists.

## Limitations

- AI-reviewed annotations are **provisional**, not independently human verified; Canteen and PUST still have QC caveats.
- No class 3 or class 4 objects are annotated and bottle data is too sparse for meaningful evaluation.
- Grouped train/validation/test splits reduce known leakage but do not certify independent field generalization.
- **No model has been trained or tested yet** as part of this GitHub upload.
