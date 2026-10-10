# Cafeteria image dataset

This directory is organized so image data can be versioned in GitHub alongside the code.

- `raw/<source>/images/`: byte-identical **source-photo snapshots** for the 2,004 images included in this delivery. These preserve original photo bytes before label remapping and review. They are **not** complete, unfiltered Roboflow exports; rejected upstream examples are not included. Original filtered LVIS train/val annotations are in `raw/lvis_plate/annotations/`.
- `processed/<source>/images/` and `labels/`: the 1,960 currently processed YOLO image/label pairs. This includes 344 provisionally included LVIS plate images (834 plate boxes). Manual review for all other visible classes remains outstanding.
- `candidates/lvis_plate/images/` and `labels/`: 44 held LVIS plate examples (293 existing plate boxes) flagged for potentially incomplete plate labeling. **Do not train on these until corrected.**
- `manifests/`: provenance and source splitting metadata.
- `audits/current/`: structural checks for the 1,960 processed images.

The processed collection has 4,658 annotations and currently lacks accepted `paper_boat`, `compartment_container`, and `napkin` examples. Structural checks do not prove label completeness. Keep final NHS test scenes independent and untouched.

**Avoid duplicates in your working copy:** `raw/` contains the same photo pixels as the corresponding `processed/` or `candidates/` path. Only `processed/` should be used for prepared YOLO training inputs. Do not point training to `raw/` and `processed/` simultaneously.

Sources and rights are described in `datasets.csv`, source READMEs and manifests. Original COCO photos retain their individual source terms.
