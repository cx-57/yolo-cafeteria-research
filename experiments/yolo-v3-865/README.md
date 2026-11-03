# YOLO V3 — 865-photo recovered utensil/wrapper supplement

**New experimental candidate, not a replacement for frozen V1 or the trained 704-photo V2.** The original 704 photos retain their source-group-safe split. Older source photos were added **only to the new training split**.

## What was recovered and changed

| Source | Photos | New utensil boxes | New wrapper boxes |
|---|---:|---:|---:|
| Original 704 dataset (unchanged photos) | 704 | 1 manually corrected fork | 0 |
| Legacy October 2026 `YOLO_Day1_Complete.zip`: utensil | 128 | 244 reused boxes | 0 |
| Legacy October 2026 `YOLO_Day1_Complete.zip`: trash detection | 33 | 0 | 37 reused boxes |
| **V3 total photos** | **865** | **621 total utensil** | **140 total wrapper** |

**Four bottle labels from V2 were removed** as requested (all photographs retained; no bottle objects annotated in V3). The seven-class indexing stays unchanged for compatibility.

- **865 photos:** 697 train, 97 validation, 71 test; 523 source/capture groups
- **4,157 YOLO boxes:** 2,266 food, 1,130 plate, 621 utensil, 140 wrapper, 0 hinged container, 0 drink carton, 0 bottle
- **One new individually verified annotation:** a fork at the right edge of `canteen__test__0_jpeg.rf.89a556b16e77e12c869ffa92f7b120f3.jpg` (original 828×823 photo); box pixels `(765, 224, 827, 451)`.
- **No duplicate image bytes** between recovered photos and the 704-image set. A basic 64-bit dHash screen found no <=6 Hamming-distance matches between these 161 recovered images and V2 val/test. This is *not* proof against every crop or near-duplicate.
- The recovered annotations are original source target-class YOLO boxes: legacy class 3→V3 class 2 (utensil), and legacy class 9→V3 class 5 (wrapper).

### Annotation caveat — important

**Do not claim the 865 images have been fully reannotated or human verified.** The 244+37 recovered boxes already existed in the old source files. The fork is the *only* new box added by visual correction here. All 704 images and each legacy image still require an exhaustive original-resolution audit for missing items, particularly source images containing other unlabeled classes. Prior V2 box totals are provisional. The original *later* V1 working corpus had 656 utensil and 476 wrapper boxes across its training/validation/test records, but some training instances were augmented views, and that complete later corpus is not included here.

## Dataset archive

File: **`YOLO_V3_865_legacy_supplement.zip`** (151 MB compressed).

SHA-256:
```text
5138749bbed039886bab1f3ab961ae9430d76a8c36ba7b8e4e57a4f910b4139a
```

The archive is available in the chat attachment, **not on GitHub**. Extract it into this folder. It contains `yolo_v3_865_legacy/` with all images, YOLO labels, `source_manifest.json` (source identity, image and label SHA-256, original split), `data.yaml`, and `build_report.json`. See `split_summary.json` in this folder.

## Train from VS Code on Apple Silicon

```bash
cd experiments/yolo-v3-865
unzip YOLO_V3_865_legacy_supplement.zip
python3 -m pip install -r requirements.txt
python3 train_yolo_v3.py --verify-only
python3 train_yolo_v3.py --device mps --epochs 20 --batch 4
```

The script validates the entire manifest and refuses to overwrite an existing results folder. A new YOLO11n model trains from pretrained `yolo11n.pt`, never from your frozen V1 or prior V2 weights.

Outputs: `YOLO_V3_trial_results/train/weights/best.pt`, training plots/CSV, and `YOLO_V3_trial_results/test_summary.json`. **No V3 model has been trained yet.**

## Google Colab

[Launch the V3 Colab training notebook](https://colab.research.google.com/github/cx-57/yolo-cafeteria-research/blob/main/experiments/yolo-v3-865/YOLO_V3_865_Colab_Train.ipynb), select a T4 GPU, then upload the *V3* ZIP. The notebook checks its archive SHA-256 and dataset membership before 20-epoch training.

## Research integrity / source attribution

- Legacy Utensil dataset: [Roboflow Utensil v1](https://universe.roboflow.com/lemon-dataset-h2ime/utensil/dataset/1), recorded as CC BY 4.0 in the recovered Day1 manifest.
- Legacy wrapper photos: [Trash Detection v1](https://universe.roboflow.com/blues-workspace-cpbsa/trash-detection-1fjjc-ntyvc/dataset/1), recorded as CC BY 4.0 in the recovered Day1 manifest.
- Other source photos: FLIC, Canteen Food Waste, PUST; check their separate source distribution terms before publicly rehosting images.

The original V1 frozen specialist checkpoints and V2 704-photo trained results were not changed. The V2 test set is historically exposed; the V3 one-fork label repair means its metrics are **not directly identical-label comparisons** with V2. Avoid calling either an independent prospective school field evaluation.
