# Dataset organization

- `data/raw/<source>/`: original source-photo snapshots and some historical Roboflow exports; **never train directly on these**.
- `data/processed/<source>/images/` and `labels/`: the **2,022 curated/provisional image–label pairs** including **62 TartanHacks frames** and **344 provisional LVIS plates**. Labels use the **retained historical 11-class IDs** and are converted on export.
- `data/candidates/plate/`: **44 held** LVIS images with potentially incomplete plate annotations; do not train on these yet.
- `data/manifests/`: historical source splits, checksum provenance, capture groups and historical class IDs.
- `data/training/`: generated, ignored, safe-to-delete working copy with **current 11-class IDs** and grouped train/validation splits.
- `data/audits/old_11class_snapshot/`: old audit results for reference; regenerate `data/audits/current/` locally.

Current processed corpus: **2,353 images, 5,063 annotated objects**, before a complete multi-class visual label review. Missing class: **paper_boat**. Public napkin frames come from a few clips and must stay grouped by capture.

Duplicate, unfiltered split directories were removed from `data/processed` for clarity; their originals remain in `data/raw` and prior Git commits. The deprecated compartment-container class had **zero accepted images** and is no longer part of the model.

Do not combine `raw`, `processed`, and `training` as one dataset; many entries contain identical images. For reliable results, retain an independent set of natural NHS test scenes from different sessions/days. Data and licenses: `data/datasets.csv`.

Optional reviewed ZIP `Waste_Detection_v3_NHS_Untrimmed_331.zip` adds 331 training-only pairs, including 75 disposable bowls for new class ID 10. These 331 images are committed under `data/processed/waste_detection_v3`; old processed sources remain on their historical label scheme and are remapped only when creating training copies.


Optional Dry & Wet Waste import: `Dry_Wet_Waste_NHS_233.zip` contains **233 training-only candidates** (143 bottles, 56 wrappers, 34 napkins). Import with `bash upload_dry_wet_waste.sh`; the other 1,593 source photos are excluded or remain held. Do not place unreviewed exports under `data/processed`. The source images are not necessarily cleared for public redistribution.

Food additions (after local import): `school_lunch_asian/` — 130 new Asian lunch scenes, and `food_american/` — 380 US-style food photos. Both use current class IDs (0 food and, for Asian set, 6 milk carton), are training-only, and must NOT be remapped as older source IDs. Import with `bash upload_food_additions.sh`. Existing School Lunch 467 images were excluded from the new selections.
