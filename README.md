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

The source list is in `data/datasets.csv`. It was intentionally reduced to avoid redundant, weak-domain, hard-to-license, or awkward-to-download datasets.

## Prepare a downloaded dataset

Export/download a source in YOLO format and place it under `data/raw/`.

Example:

```bash
python3 src/prepare.py data/raw/school_lunch school_lunch
```

`prepare.py` does the mechanical cleanup:
- keeps only approved source classes
- remaps them to our final class IDs
- validates YOLO boxes
- removes exact duplicates
- hard-links retained images when possible, so raw + processed do not use two copies of the image bytes
- prints the kept counts

It does **not** decide whether an individual photo visually matches NHS. Every retained candidate still needs visual review before final training.

After a source has been reviewed and the retained files are safe in `data/processed/`, its raw download can be deleted to save space. Hard-linked processed images remain on disk.

## Augmentation

Augmentation happens during training only. It does not create stored image copies and does not count as new data.

`src/train.py` uses mild rotation, translation, scaling, horizontal flips, lighting/color changes, and light mosaic. Validation and final-test images are never augmented.

## Files

```text
config.yaml
data/datasets.csv
src/prepare.py
src/train.py
```

That is the intended core of the project. Keep it simple.

## Workflow

1. Download one listed source in YOLO format.
2. Run `prepare.py`.
3. Visually review every candidate image against the NHS standard.
4. Delete rejected candidates.
5. Delete the raw source after review to save space.
6. Fill remaining class deficits with staged NHS photos.
7. Freeze the natural NHS final test set.
8. Train and evaluate.
