# YOLO Cafeteria Research

Object detection for Natick High School-style cafeteria items. The model predicts **items**, not whether they belong in recycling, compost or trash; school-specific disposal rules remain a separate step.

## Current state (October 11, 2026)

**3,096 processed original image/label pairs** across 12 sources on GitHub. These are public/provisional source images, **not** 3,096 independent NHS cafeteria scenes. Some annotations may omit other visible objects. Do not combine processed images with raw copies or generated training views.

The active detection IDs in [config.yaml](config.yaml) are:

| ID | Class | ID | Class |
|---:|---|---:|---|
| 0 | food | 6 | drink_carton |
| 1 | plate | 7 | condiment_packet |
| 2 | paper_boat | 8 | wrapper |
| 3 | utensil | 9 | bottle |
| 4 | hinged_container | 10 | bowl |
| 5 | napkin | | |

**Important limitations:** No curated paper-boat annotations yet. Condiments have only 44 independent originals. Bowl examples and several newly imported sources are training-only because source train/validation examples overlap. Some validation classes will therefore have no measurable AP. This is appropriate: don't manufacture validation performance by leaking related images.

## Next-phase workflow

### 1. Install and check

Run from the cloned repository:

    git pull
    python3 -m venv .venv
    source .venv/bin/activate
    python3 -m pip install -r requirements.txt
    python3 -m unittest discover -s tests -v
    python3 src/audit.py --manifests data/manifests --output data/audits/current

The audit is read-only for originals, writes a local report, and may report source-label limitations that need manual review. Inspect it before describing the data as fully clean. You can also check local disk space using \`df -h .\`.

### 2. Prepare the **10x** dataset without training

    python3 src/train.py --build-only --augment-factor 10

This runs the original-label / image-decoding checks, groups Roboflow variants, keeps exact-pixel duplicates across different sources in one split, and applies a deterministic 80/20 group-level target where independent public-source splits are feasible. Condiments, TartanHacks video and four new potentially correlated imports stay **training-only**.

**Precisely what 10x means:** Every training original gets one original file path and **nine additional hardlinked training-view paths**. Each view is independently and randomly augmented by Ultralytics during training (brightness, color, mild geometry, horizontal flipping and mosaic). There are **ten training views per original** but **not ten independently photographed examples**, and not nine synthetic image files stored on disk. Hardlinks save considerable disk space. Validation contains exactly one untouched copy per held-out original and is not augmented. Do **not** commit \`data/training/\` to GitHub.

Inspect \`data/training/split_summary.json\` for the actual train/validation counts, original per-class box/image support, and classes absent from validation.

### 3. Train YOLO11n

To use the prepared dataset without rebuilding it:

    python3 src/train.py data/training/data.yaml --device mps --model yolo11n.pt --epochs 100 --batch 8 --name yolo11n-10x

\`--device mps\` is for **Apple Silicon**. For an NVIDIA/CUDA GPU use \`--device 0\`. If you lack an accelerator, omit \`--device\` for auto-detection, but CPU training will be slower. Reduce \`--batch\` to 4 or 2 if you run out of memory.

Training uses 640-pixel images, reproducible seed 42, early stopping with patience 25, and saves checkpoints and learning curves under \`runs/yolo11n-10x/\`. After training, the script evaluates \`best.pt\` against the **unchanged provisional validation** set, saves AP/precision/recall metrics and per-class results (unmeasurable classes are \`null\`), plus plots. YOLO randomly transforms the hardlinked views; it does not transform validation originals.

To build and train in one command instead, omit the positional data.yaml:

    python3 src/train.py --augment-factor 10 --device mps --name yolo11n-10x

To resume a stopped run (the existing matching \`data/training/\` must be preserved):

    python3 src/train.py --resume runs/yolo11n-10x/weights/last.pt

To re-evaluate an existing model on the currently prepared validation set:

    python3 src/train.py --evaluate runs/yolo11n-10x/weights/best.pt --device mps

### 4. Examine errors, not only aggregate mAP

Inspect \`runs/yolo11n-10x/results.csv\`, \`confusion_matrix.png\`, training/validation example plots and the generated \`evaluation/metrics.json\`. In particular check false positives from incomplete labels, confusing plastic vs paper objects, occlusion, small packets, distant utensils, the school-specific objects, and class coverage. **A class with no independently held-out validation examples cannot be rated.**

### 5. Compare against a 1x control

A fair research ablation keeps the same seed and original-image group split. Run:

    python3 src/train.py --augment-factor 1 --device mps --name yolo11n-1x

This rebuilds the Git-ignored working dataset with one training view per original. Complete and archive the 10x results before doing so. Both models retain the same **original** validation split; augmentation does **not** change the number of genuinely different original photos. Record run settings, class-wise results, and failure examples for both. Do not conclude that 10x is better without measuring.

### 6. Collect a truly independent NHS test set before research claims

Take and annotate new natural NHS cafeteria scenes from separate days/lighting/cameras, with **all visible target-class instances labeled**, and keep them entirely outside train/validation and augmentation. Record capture groups and licenses/consent appropriately. Evaluate the frozen best model on these images once, reporting per-class precision, recall, mAP50 and mAP50-95, along with support and error examples. Acquire paper-boat originals and independent examples of packets, bowls and napkins if these classes matter in practice.

**No actual model training runs automatically on GitHub:** the repository contains scripts and CPU-only tests. Training requires a local or cloud machine with Python/PyTorch and preferably GPU/MPS acceleration. GitHub Actions checks synthetic preprocessing tests; it is not a GPU trainer.

## Repository structure

- \`config.yaml\` — active 11 detection classes
- \`src/prepare.py\` — Roboflow original-label conversion for review candidates
- \`src/lvis.py\` — real LVIS plate image/box candidate importer; flagged photos remain held
- \`src/audit.py\` — original-corpus data/provenance and annotation audits
- \`src/train.py\` — complete build → 10x views → train → validation pipeline
- \`tests/\` — quick synthetic label and leakage tests
- \`data/raw/\` — originals and historical source material; not used directly for training
- \`data/processed/\` — the 3,096 retained original image/label pairs
- \`data/manifests/\` — sources, checksums and image provenance
- \`data/audits/\` — historical audit evidence and known caveats
- \`data/training/\` — regenerated, ignored training views
- \`runs/\` — ignored model checkpoints and evaluation artifacts

## Historical IDs — DO NOT remap the original stored files

Previously stored sources use original 11-class IDs; the training builder converts only its generated labels. Historical IDs 0, 1, 2, 3 stay 0, 1, 2, 3; old 5→4 (hinged_container), 6→5 (napkin), 7→6 (drink_carton), 8→7 (condiment_packet), 9→8 (wrapper), 10→9 (bottle); old 4 is retired. Newly imported training-only sources already use the current IDs, including new 10=bowl. Never blend current-ID and historical-ID labels in the same processed source folder.

See [dataset provenance](data/datasets.csv), [LVIS review](data/audits/lvis_2026-10-10/README.md), [waste detection screening](data/audits/waste_detection_v3_2026-10-10.md), [food additions](data/audits/food_additions_2026-10-11.md) and [raw/processed organization](data/README.md).
