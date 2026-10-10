# Cafeteria image dataset

The datasets use this layout:

- `data/raw/<source>/images/`: byte-identical source-photo snapshots of the images that were retained or held for review. This is **not** every image in the original unfiltered Roboflow exports. For LVIS, original filtered annotation JSON is also stored under `raw/lvis_plate/annotations/`.
- `data/processed/<source>/images/` and `labels/`: reviewed or provisionally accepted YOLO image/label pairs. Current prepared delivery includes 1,960 images and 4,658 boxes; LVIS plate images are provisional until all visible classes are checked.
- `data/candidates/lvis_plate/`: 44 LVIS images with potentially incomplete plate labels, held out until fixed.
- `data/manifests/`: provenance and source split metadata.
- `data/audits/`: structural validation reports.

Only include `data/processed` in training inputs. The same original photo pixels also appear under `data/raw`; do not train on both copies or the held candidates. No natural NHS test set is included. Original image rights and dataset sources are documented in `data/datasets.csv`.
