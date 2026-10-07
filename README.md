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
- validates YOLO boxes
- removes exact duplicate images
- merges useful labels when the same exact image appears in multiple sources
- hard-links retained images when possible so raw + processed do not store two copies
- prints the kept counts

It does **not** decide whether an individual photo visually matches NHS. Every candidate must still be visually reviewed before final training.

After review, the raw source can be deleted to save space. Deleting the raw hard link does not delete the accepted processed image.

## Augmentation

Augmentation happens during training only. It creates no stored image copies and does not count as new data.

`src/train.py` uses mild rotation, translation, scaling, horizontal flips, lighting/color changes, and light mosaic. Validation and final-test images are never augmented.

## Core files

```text
config.yaml
data/datasets.csv
src/prepare.py
src/train.py
```

## Workflow

1. Download one listed source in YOLO format.
2. Run `prepare.py`.
3. Visually review every candidate image against the NHS standard.
4. Remove mismatched image/label pairs.
5. Commit only the accepted image/label pairs under `data/processed/<source_key>/`.
6. Delete the raw source after review.
7. Fill remaining deficits with staged NHS photos.
8. Freeze the natural NHS final test set.
9. Train and evaluate.
