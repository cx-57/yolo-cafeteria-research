# YOLO Cafeteria Research

Detect NHS-style cafeteria items using **10 object classes** with Ultralytics YOLO11. Recycling/compost/trash decisions are **separate school-specific rules**, not predictions made by object detection.

## Files

| File | Purpose |
|---|---|
| `config.yaml` | Current IDs for the 10 classes |
| `src/prepare.py` | Convert Roboflow exports to review candidates |
| `src/lvis.py` | Download/verify real LVIS paper-plate photos and boxes |
| `src/audit.py` | Audit annotation integrity and dataset provenance |
| `src/train.py` | Build grouped train/validation data and train YOLO11 |
| `tests/` | Small checks for class conversion, geometry and auditing |

Only Ultralytics, PyYAML and Pillow are third-party dependencies.

## Current object classes

| ID | Class | Images containing class (Oct 10) |
|---:|---|---:|
| 0 | food | 467 |
| 1 | plate | 344 (provisional) |
| 2 | paper_boat | 0 |
| 3 | utensil | 155 |
| 4 | hinged_container | 121 |
| 5 | napkin | 35 (provisional) |
| 6 | drink_carton | 1,099 |
| 7 | condiment_packet | 44 |
| 8 | wrapper | 33 |
| 9 | bottle | 168 |

Images can contain multiple classes. **2,022 curated/provisional image-label pairs** with **4,721 boxes** are stored as source-specific flat pairs under `data/processed/<source>/images/` and `labels/`. This does *not* mean every image has complete multiclass labeling. Paper boats need data. New natural NHS scenes are required for final testing.

The school **does not use a compartment container**. That unused class was removed; there were zero curated images of it. Old unfiltered Roboflow export copies were removed from `data/processed`, while source data under `data/raw` and Git history remain.

## Important: historical label IDs

The already-committed `data/processed/` annotations retain their **original 11-class numeric IDs** for provenance. The active **10-class mapping** is defined in `config.yaml`. `src/train.py` converts original IDs **in the generated training copy**; `src/audit.py` uses the same conversion. Do **not** point YOLO directly at the historical raw/processed labels as a 10-class dataset.

| Old stored ID | Current model ID |
|---|---|
| 0 food | 0 |
| 1 plate | 1 |
| 2 paper_boat | 2 |
| 3 utensil | 3 |
| 5 hinged_container | 4 |
| 6 napkin | 5 |
| 7 drink_carton | 6 |
| 8 condiment_packet | 7 |
| 9 wrapper | 8 |
| 10 bottle | 9 |

Original ID 4 is retired. The stored source manifests likewise preserve historical numeric IDs. If you accept new candidate labels made with the current mapping, put them under a **new source folder name** rather than mixing them into an existing historical-label folder.

## Workflows

1. **Import** an export: `python3 src/prepare.py data/raw/school_lunch school_lunch`. It goes into `data/candidates` for review; it is not automatically accepted.
2. **Add LVIS plates** (if needed): `python3 src/lvis.py`. It fetches real COCO photos referenced by LVIS; 44 flagged candidate images are held for review. The 344 provisionally included plates need a full annotation review for other visible objects.
3. **Check data**: `python3 -m unittest discover -s tests -v` and `python3 src/audit.py --manifests data/manifests --output data/audits/current`. `data/audits/old_11class_snapshot` keeps the historical report; regenerate `current` locally for the new classes.
4. **Prepare training data without training**: `python3 src/train.py --build-only`. This builds `data/training/` with converted **10-class** labels and deterministic grouped train/validation splits (80/20 target). Condiment views on the same backdrop stay in training, and TartanHacks video frames are grouped by clip. The generated folder is Git-ignored.
5. **Train**: `python3 src/train.py` (100 epochs by default), or supply your own already-reviewed `data.yaml` with the exact 10 names/IDs.

**Research limitation:** these are provisional public-source training/validation splits. Do not treat them as a trustworthy final NHS generalization estimate: source scenes may be correlated, the condiment group has no independent public-source validation, and some labels lack complete multiclass review. Reserve completely separate natural NHS cafeteria scenes/days for final evaluation.

Sources, licenses and manifests are in `data/datasets.csv` and `data/manifests/`. Raw original images and historical annotation IDs are maintained as provenance.

## Optional waste-detection-v3 training subset (October 10)

A three-check audit screened **all 923** images (structure/boxes, NHS-domain visual contact sheets, video-frame duplicates and train/validation leakage). **250** frames showed compatible target objects, but **199** were held as near-adjacent video frames; **51** training candidates were retained: 2 plate, 9 utensil, 7 napkin, 24 wrapper, 9 bottle. **673** were rejected. The original Roboflow 877/46 train/valid split is **not independent**: four source videos cross both sets. The kept images have converted **current 10-class IDs**; the other processed sources still use historical IDs. The detailed results are in [the waste-detection-v3 audit](data/audits/waste_detection_v3_2026-10-10.md).

To add the already-reviewed `Waste_Detection_v3_NHS_Curated.zip` package from the audit download, run at the repository root:

```bash
mkdir -p data/processed/waste_detection_v3
unzip -oq ~/Downloads/Waste_Detection_v3_NHS_Curated.zip -d data/processed/waste_detection_v3
cp data/processed/waste_detection_v3/manifest.csv data/manifests/waste_detection_v3.csv
python3 src/train.py --build-only
```

The build includes these new images **only in training**, not validation. Audit the final merged corpus and do not treat frames from these demonstration clips as independent NHS cafeteria evaluation. The ZIP includes `images/`, `labels/`, `manifest.csv`, and source attribution; the source images are not yet tracked on GitHub until imported and committed separately.
