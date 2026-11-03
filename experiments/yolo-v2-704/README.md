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
2. Clone the GitHub repository if necessary, and put the ZIP in this folder (`experiments/yolo-v2-704/`).
3. From Terminal, run:

```bash
cd /path/to/yolo-cafeteria-research/experiments/yolo-v2-704
unzip YOLO_V2_704_GroupSplit_Trial.zip
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 train_yolo_v2_trial.py --verify-only
python3 train_yolo_v2_trial.py --device mps --epochs 35 --batch 4
```

The dataset extracts into `yolo_v2_trial_704/` (gitignored). The archive also includes its own training script if you prefer to train from the extracted folder independently.

The experiment initializes **new `yolo11n.pt` pretrained weights** (automatically downloaded by Ultralytics) and saves new weights as `YOLO_V2_trial_results/train/weights/best.pt`. Final metrics are saved to `YOLO_V2_trial_results/test_summary.json`, including test mAP50, mAP50-95, precision, recall, and per-supported-class AP.

## Colab GPU

Use [`YOLO_V2_704_Colab_Train.ipynb`](YOLO_V2_704_Colab_Train.ipynb), select a GPU runtime, and upload the same ZIP. This does not overwrite the frozen specialists.

## Limitations

- AI-reviewed annotations are **provisional**, not independently human verified; Canteen and PUST still have QC caveats.
- No class 3 or class 4 objects are annotated and bottle data is too sparse for meaningful evaluation.
- Grouped train/validation/test splits reduce known leakage but do not certify independent field generalization.
- **No model has been trained or tested yet** as part of this GitHub upload.
