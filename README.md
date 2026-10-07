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

## NHS visual standard

Public data is only kept when the actual object looks like something NHS uses. A matching class name is not enough.

| Class | Keep | Reject |
|---|---|---|
| food | food NHS plausibly serves | unrelated cuisine just because it is food |
| plate | disposable 9-inch sugarcane/fiber-style plate | ceramic or reusable plates |
| paper_boat | open cafeteria paper food tray; especially red-plaid style | pizza boxes, closed paper boxes |
| utensil | black or white disposable plastic fork/spoon/knife | metal silverware, chopsticks |
| compartment_container | clear cafeteria compartment container or molded-fiber compartment tray | household food-storage containers |
| hinged_container | clear plastic or fiber/bagasse food clamshell | household Tupperware |
| napkin | disposable cafeteria napkin; especially natural/brown | cloth napkins or unrelated tissues |
| drink_carton | small aseptic juice/milk-box carton | large cartons that do not resemble NHS drink boxes |
| condiment_packet | individual flat ketchup/mayo/ranch/BBQ/mustard/dressing-style packet | random sachets or non-food packets |
| wrapper | NHS-like snack/food wrapper | unrelated plastic film or packaging |
| bottle | beverage/water bottle | shampoo, detergent, chemical bottles |

NHS reference objects include red-plaid paper food trays, black/white plastic utensils, natural Tork-style napkins, clear 4-compartment containers, molded-fiber 5-compartment trays, clear 6/8-inch hinged containers, bagasse clamshells, small aseptic juice boxes, individual condiment packets, snack wrappers, and beverage bottles.

## Dataset rule

The finalized source list is in `data/datasets.csv`.

For every source:

1. Filter to potentially useful source classes.
2. Manually inspect every candidate image that would be retained.
3. Reject images that do not match the NHS visual standard above.
4. Remap the accepted labels to our classes.
5. Deduplicate.
6. Count the accepted originals.

`target_images` in `data/datasets.csv` is a planning goal, not a quota. Never keep a bad match just to hit the target.

TACO and RTWFD are auxiliary only. They should not be used to pad class counts.

Raw data stays local in:

```text
data/raw/
```

Cleaned data goes in:

```text
data/processed/
```

Both are ignored by Git.

## Augmentation

Use augmentation during training only, after the original images are cleaned and split.

- augment training images only
- never augment validation or final-test images
- augmented views do not count as new source images
- keep transformations realistic
- no vertical flips
- no heavy perspective/shear
- no synthetic images just to inflate weak classes

The default settings in `src/train.py` use mild rotation, translation, scale, horizontal flips, lighting/color variation, and light mosaic.

## Files

- `config.yaml` — final classes and image targets
- `data/datasets.csv` — final public/local source plan
- `src/audit_dataset.py` — counts images, labels, classes, and bad YOLO rows
- `src/remap_classes.py` — remaps source class IDs
- `src/deduplicate.py` — reports duplicate images
- `src/train.py` — trains YOLO with conservative train-time augmentation

## Current step

1. Download one source at a time.
2. Filter its source classes.
3. Visually review every image we plan to keep.
4. Remap and deduplicate.
5. Fill remaining deficits with real NHS photos.
6. Freeze the natural NHS final test set.
7. Train and evaluate.
