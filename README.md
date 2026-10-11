# YOLO Cafeteria Research

Detect NHS-style cafeteria items using **11 object classes** with Ultralytics YOLO11. Recycling/compost/trash decisions are **separate school-specific rules**, not predictions made by object detection.

## Files

| File | Purpose |
|---|---|
| `config.yaml` | Current IDs for the 11 classes |
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
| 10 | bowl | 0 (new; 75 more photos packaged for import) |

Images can contain multiple classes. **2,022 curated/provisional image-label pairs** with **4,721 boxes** are stored as source-specific flat pairs under `data/processed/<source>/images/` and `labels/`. This does *not* mean every image has complete multiclass labeling. Paper boats need data. New natural NHS scenes are required for final testing.

The school **does not use a compartment container**. That unused class was removed; there were zero curated images of it. Old unfiltered Roboflow export copies were removed from `data/processed`, while source data under `data/raw` and Git history remain.

## Important: historical label IDs

The already-committed `data/processed/` annotations retain their **original 11-class numeric IDs** for provenance. The active **11-class mapping** is defined in `config.yaml`. `src/train.py` converts original IDs **in the generated training copy**; `src/audit.py` uses the same conversion. Do **not** point YOLO directly at the historical raw/processed labels as an 11-class dataset.

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
4. **Prepare training data without training**: `python3 src/train.py --build-only`. This builds `data/training/` with converted **11-class** labels and deterministic grouped train/validation splits (80/20 target). Condiment views on the same backdrop stay in training, and TartanHacks video frames are grouped by clip. The generated folder is Git-ignored.
5. **Train**: `python3 src/train.py` (100 epochs by default), or supply your own already-reviewed `data.yaml` with the exact 11 names/IDs.

**Research limitation:** these are provisional public-source training/validation splits. Do not treat them as a trustworthy final NHS generalization estimate: source scenes may be correlated, the condiment group has no independent public-source validation, and some labels lack complete multiclass review. Reserve completely separate natural NHS cafeteria scenes/days for final evaluation.

Sources, licenses and manifests are in `data/datasets.csv` and `data/manifests/`. Raw original images and historical annotation IDs are maintained as provenance.

## Waste Detection v3 (expanded NHS bowls)

The user confirmed the source's **white disposable food bowls and kraft paper soup tubs are used at NHS**. A revised three-check review retains **all 250** previously eligible frames (including 199 near-adjacent frames), plus **75** newly accepted disposable bowls and **6** disposable plates. **331 training-only images / 342 boxes** in total. Nothing was thinned.

The **new class is `10: bowl`**, appended without changing IDs 0–9. It covers disposable cafeteria food bowls/soup tubs, not glass bowls or beverage cups. The original Roboflow train/validation groups overlap: source images **must not be used as independent validation/test examples**. The generated training split keeps this entire source in training.

To upload the **exact audited 331 image/label pairs** from your Mac to GitHub, download `Waste_Detection_v3_NHS_Untrimmed_331.zip` from ChatGPT to `~/Downloads`. Then run this single command in your cloned repository (GitHub authentication must already be configured):

```bash
bash upload_waste_detection_v3.sh
```

The uploader checks image/label matching, 342 valid boxes including 82 bowls, the provenance manifest, and conflicting files before committing to `main`. The original labels are already in the **current 11-class ID format**. The training builder will include these images **only in training**.

**Do not merge with the obsolete 51-image subset**: the 331-image ZIP replaces it, and includes those 51 already. JPEGs and their converted YOLO labels are distributed in the ZIP, not stored on GitHub yet. For detailed decisions and limitations see [the updated three-check audit](data/audits/waste_detection_v3_2026-10-10.md).
