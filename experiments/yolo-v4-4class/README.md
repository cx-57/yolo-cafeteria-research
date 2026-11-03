# YOLO V4 — four-class source-mix experiment

**Experimental only. Not a completed annotation audit or independent benchmark.**

## Full dataset and source controls

Download `YOLO_V4_865_4class_audit_pending.zip` from the ChatGPT attachment (151,005,381 bytes), SHA256 `95a386dbd9cf707e642237b92326a27ffd4958b893d1e0a7089e463b823c91e7`.

The complete ZIP includes 865 original source image files, each with a 4-class YOLO label text file, `data.yaml`, `data_supplemented.yaml`, source hash manifest, `audit_queue.csv`, `audit_summary.json`, annotation policy, review overlay script, and standalone Python trainer. Source images are **not stored in GitHub**. The GitHub notebook uses the script shipped **inside the ZIP**.

- Primary cafeteria-only train: **521** images, boxes: food 1,749, plate 849, utensil 294, wrapper 79.
- Optional legacy-supplemented train: **682** images, boxes: food 1,749, plate 849, utensil 538, wrapper 116.
- Validation: **97** images, boxes: food 274, plate 163, utensil 48, wrapper 13.
- Test: **71** images, boxes: food 239, plate 117, utensil 35, wrapper 11.
- Quarantine: **15** source images including **one visually confirmed train/test near-duplicate** and 14 ambiguous unlabeled PUST soup/liquid bowl photos.
- The original validation/test **image membership** and four-class target box geometry were preserved from V3. Nine empty-label val photos and one empty-label test photo require adjudication.

A name-preserving dHash screen of remaining photos found no train/held-out matches at Hamming distance ≤6 after the confirmed duplicate was quarantined. That is not proof of no exposure or leakage. There is **no claim of exhaustive original-resolution manual review**; all 865 entries in `audit_queue.csv` are flagged `NOT_REVIEWED`. Some PUST source images are already resized to 640×640.

## Train

[Open the Google Colab V4 notebook](https://colab.research.google.com/github/cx-57/yolo-cafeteria-research/blob/main/experiments/yolo-v4-4class/YOLO_V4_4Class_Colab_Train.ipynb).

Select T4, Run all, upload V4 ZIP. It trains **YOLO11s**, 800px, max 80 epochs, patience 20, using the 521 cafeteria training images by default. The optional 161-photo legacy ablation is separately selectable (set `RUN_SUPPLEMENTED=True`), with the **same** validation/test membership.

**Test evaluation is OFF by default**. Select the winning candidate using validation scores and review/freeze test annotations before evaluating the test exactly once.

The frozen original V1 specialists and trained V2/V3 experiments are unchanged. The released V4 artifact contains no pretrained or newly trained V4 weights. The test set remains historically exposed to earlier work; do not call it an independent real-world field cohort.

For reproduction offline, extract the data ZIP into the folder containing `train_yolo_v4.py`, then run:

```bash
python train_yolo_v4.py verify --root .
python train_yolo_v4.py train --root . --variant cafeteria --output ./V4_cafeteria --device mps --epochs 80 --patience 20 --imgsz 800 --batch 4
```

On Colab, use `--device 0` as already configured by the notebook.
