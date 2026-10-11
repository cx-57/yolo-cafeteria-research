# Dataset organization

- `data/raw/<source>/`: original source-photo snapshots and some historical Roboflow exports; **never train directly on these**.
- `data/processed/<source>/images/` and `labels/`: the **3,096 curated/provisional original image–label pairs** across 12 sources. Older source labels use historical class IDs, whereas recent imported sources already use current 11-class IDs; only older source labels are remapped in the generated training copy.
- `data/candidates/plate/`: **44 held** LVIS images with potentially incomplete plate annotations; do not train on these yet.
- `data/manifests/`: historical source splits, checksum provenance, capture groups and historical class IDs.
- `data/training/`: generated, ignored, safe-to-delete working copy with **current 11-class IDs** and grouped train/validation splits.
- `data/audits/old_11class_snapshot/`: old audit results for reference; regenerate `data/audits/current/` locally.

Current processed corpus: **3,096 original images**, before a complete multi-class visual label review. Regenerate the current audit for up-to-date object counts. Missing class: **paper_boat**. Public napkin frames come from a few clips and must stay grouped by capture.

Duplicate, unfiltered split directories were removed from `data/processed` for clarity; their originals remain in `data/raw` and prior Git commits. The deprecated compartment-container class had **zero accepted images** and is no longer part of the model.

Do not combine `raw`, `processed`, and `training` as one dataset; many entries contain identical images. For reliable results, retain an independent set of natural NHS test scenes from different sessions/days. Data and licenses: `data/datasets.csv`.

The committed Waste Detection v3 subset contributes 331 training-only pairs, including 75 disposable bowls for new class ID 10. These 331 images are committed under `data/processed/waste_detection_v3`; old processed sources remain on their historical label scheme and are remapped only when creating training copies.


The committed Dry & Wet Waste subset contains **233 training-only candidates** (143 bottles, 56 wrappers, 34 napkins). The other 1,593 source photos are excluded or remain held. Do not place unreviewed exports under `data/processed`. The source images are not necessarily cleared for public redistribution.

Committed food additions: `school_lunch_asian/` — 130 new Asian lunch scenes, and `food_american/` — 380 US-style food photos. Both use current class IDs (0 food and, for Asian set, 6 milk carton), are training-only, and must NOT be remapped as older source IDs. They are already present in `data/processed/`. Existing School Lunch 467 images were excluded from the new selections.

Training views: `python3 src/train.py --build-only --augment-factor 10` generates ten hardlinked **train** paths per source original, with stochastic Ultralytics augmentation during training. **Validation is never augmented**. See the main README for exact reproducible training and evaluation commands.
