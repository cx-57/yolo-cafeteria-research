# YOLO Cafeteria Research

Goal: build a YOLO system that recognizes common NHS cafeteria waste items and maps them to local disposal guidance.

## Final classes

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

The classes are based on items actually used at NHS. Public datasets are only used when the images visually match NHS food or serviceware.

Examples of NHS items we are targeting:
- red-plaid paper food boats
- disposable sugarcane/fiber plates
- molded-fiber compartment trays
- clear compartment containers
- clear plastic clamshells
- fiber/bagasse clamshells
- disposable plastic utensils
- disposable cafeteria napkins
- small aseptic drink cartons
- condiment packets
- snack/food wrappers
- beverage bottles
- NHS-like food

## Data plan

The finalized source list is in `data/datasets.csv`.

Nothing has been downloaded into the repo yet. Raw images stay local under:

```text
data/raw/
```

The cleaned YOLO dataset will eventually go under:

```text
data/processed/
```

Both folders are ignored by Git because image datasets are too large to commit.

## Important rule

Do not use an image just because its class name matches.

Keep only images that realistically resemble something seen at NHS. For example:
- disposable fiber plate: keep
- ceramic restaurant plate: reject
- plastic cafeteria fork: keep
- metal fork: reject
- clear food clamshell: keep
- household Tupperware: reject

## Files

- `config.yaml` — final class names, image targets, and disposal-rule placeholders
- `data/datasets.csv` — finalized public/local dataset sources
- `src/audit_dataset.py` — count images, labels, classes, and bad YOLO rows
- `src/remap_classes.py` — convert source class IDs into our class IDs
- `src/deduplicate.py` — report duplicate images
- `src/train.py` — simple Ultralytics YOLO training script

## Current step

1. Download the finalized datasets.
2. Keep only NHS-like images/classes.
3. Remap labels to the classes in `config.yaml`.
4. Deduplicate and audit.
5. Fill class deficits with NHS-specific photos.
6. Train the first model.
