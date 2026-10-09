# YOLO Cafeteria Research

Real-time YOLO detection of common NHS cafeteria items, followed by a separate local disposal-rule step.

## Classes

```text
food
plate
paper_boat
utensil
compartment_container
hinged_container
napkin
drink_carton
condiment_packet
wrapper
bottle
```

## Data rule

Only keep an image when the actual object looks like something NHS uses. A matching source label is not enough.

Examples:
- sugarcane/fiber disposable plate: keep; ceramic plate: reject
- open cafeteria paper boat: keep; pizza box: reject
- black/white disposable plastic utensil: keep; metal silverware: reject
- clear/fiber cafeteria container: keep; household Tupperware: reject
- natural/brown cafeteria napkin: keep; cloth napkin: reject
- small aseptic drink carton: keep; large carton: reject
- cafeteria condiment packet: keep; unrelated sachet: reject
- beverage bottle: keep; chemical/shampoo bottle: reject

The source list is in `data/datasets.csv`.

`paper_boat` is intentionally local-first because NHS uses a very specific open red-plaid tray. Condiment packets and exact NHS tray/container shapes should also rely heavily on local photos rather than generic public examples.

## Storage

Raw downloads and model outputs stay local. Manually reviewed processed subsets may be tracked in GitHub so the exact training corpus is reproducible.

- raw downloads: `data/raw/` (gitignored)
- reviewed images/labels: `data/processed/<source_key>/` (tracked)
- model runs/weights: `runs/` (gitignored)

Only visually accepted image/label pairs belong in `data/processed/`; rejected source images should never be committed.

## Prepare a downloaded dataset

Download/export a source in YOLO format and place it under `data/raw/`.

Example:

```bash
python3 src/prepare.py data/raw/school_lunch school_lunch
```

`prepare.py`:
- keeps only approved source classes
- remaps them to the final class IDs
- rejects nonfinite, zero-size, out-of-range, and out-of-image YOLO boxes
- converts Roboflow segmentation polygons to axis-aligned YOLO detection boxes when needed
- writes a flat `data/candidates/<source>/images|labels` layout, preserving the source split in the filename
- hard-links candidate images when possible, with a copy fallback across filesystems
- refuses to overwrite different existing labels or image bytes
- prints candidate counts

Exact and near-duplicate detection is performed by `src/audit.py`, not by `prepare.py`. Near matches are review candidates; they are never deleted automatically. Candidates must be reviewed before promotion to `data/processed/`.

It does **not** decide whether an individual photo visually matches NHS. Every candidate must still be visually reviewed before final training.

After review, the raw source can be deleted to save space. Deleting the raw hard link does not delete the accepted processed image.

## Visual audit records

Image-level review decisions are stored under `data/audits/`. A retained subset's source-index manifest is stored under `data/manifests/`. Rejected images never enter `data/processed/`.

## Augmentation

Augmentation happens during training only. It creates no stored image copies and does not count as new data.

`src/train.py` uses mild rotation, translation, scaling, horizontal flips, lighting/color changes, and light mosaic. Validation and final-test images are never augmented.

## Core files

```text
config.yaml
data/datasets.csv
src/prepare.py
src/audit.py
src/train.py
```

## Workflow

1. Download one listed source in YOLO format.
2. Run `prepare.py`.
3. Visually review every candidate image against the NHS standard, including every visible target object.
4. Copy accepted image/label pairs to `data/processed/<source>/images|labels`; preserve provenance in `data/manifests/`.
5. Commit only the accepted image/label pairs under `data/processed/<source_key>/`.
6. Delete the raw source after review.
7. Fill remaining deficits with staged NHS photos.
8. Freeze the natural NHS final test set.
9. Train and evaluate.

## Day 1 integrity checks

```bash
python3 -m unittest discover -s tests -v
python3 src/audit.py --manifests data/manifests
# If the original exports are available, also verify annotation preservation:
python3 src/audit.py --raw data/raw --manifests data/manifests
```

The audit writes inventory, class counts, exact/near-duplicate evidence, source annotation gaps, and a JSON summary to `data/audits/day1/`. Nonzero exit means a structural/provenance/source-preservation problem. Exit zero does **not** establish complete visual annotation or readiness to train all 11 classes.

The October 9 audit recovered **1,574 images and 3,782 instances**. After quarantining two mismatched images and adding two missing utensil boxes, the active corpus has **1,572 images and 3,780 instances**. There are no exact duplicates and **83 near-duplicate candidate pairs**. Five classes have zero annotations. The [Day 1 report](data/audits/2026-10-09_day1_report.md) explains the unresolved visual annotation and domain issues. Existing images are retained as historical reviewed candidates; this audit does not certify them as a clean 11-class training corpus.

