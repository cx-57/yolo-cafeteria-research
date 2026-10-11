# Dataset organization

- `data/raw/<source>/`: original source-photo snapshots and some historical Roboflow exports; **never train directly on these**.
- `data/processed/<source>/images/` and `labels/`: the **2,022 curated/provisional image–label pairs** including **62 TartanHacks frames** and **344 provisional LVIS plates**. Labels use the **retained historical 11-class IDs** and are converted on export.
- `data/candidates/lvis_plate/`: **44 held** LVIS images with potentially incomplete plate annotations; do not train on these yet.
- `data/manifests/`: historical source splits, checksum provenance, capture groups and historical class IDs.
- `data/training/`: generated, ignored, safe-to-delete working copy with **current 11-class IDs** and grouped train/validation splits.
- `data/audits/old_11class_snapshot/`: old audit results for reference; regenerate `data/audits/current/` locally.

Current processed corpus: **2,022 images, 4,721 annotated objects**, before a complete multi-class visual label review. Missing class: **paper_boat**. Public napkin frames come from a few clips and must stay grouped by capture.

Duplicate, unfiltered split directories were removed from `data/processed` for clarity; their originals remain in `data/raw` and prior Git commits. The deprecated compartment-container class had **zero accepted images** and is no longer part of the model.

Do not combine `raw`, `processed`, and `training` as one dataset; many entries contain identical images. For reliable results, retain an independent set of natural NHS test scenes from different sessions/days. Data and licenses: `data/datasets.csv`.

Optional reviewed ZIP `Waste_Detection_v3_NHS_Untrimmed_331.zip` adds 331 training-only pairs, including 75 disposable bowls for new class ID 10. These images are **not uploaded to GitHub yet**; old processed sources remain on their historical label scheme and are remapped only when creating training copies.
