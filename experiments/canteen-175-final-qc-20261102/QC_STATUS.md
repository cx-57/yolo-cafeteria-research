# Canteen 175-image final AI review checkpoint

Status: **QC PENDING** — not independent human verified and not yet safe for model training.

- 175/175 source images individually viewed at full native image resolution across Batches 01–06.
- Original JPEGs preserved: 175/175 SHA256 matching source manifest.
- YOLO label files: 175/175. Annotated full-resolution previews: 175/175.
- Revised YOLO boxes: 925 total: class 0 food 426, class 1 plate/bowl 229, class 2 utensil 249, class 3 hinged_container 0, class 4 drink_carton 0, class 5 wrapper 18, class 6 bottle 3.
- Batch 05: 40 images, 210 boxes. Batch 06: 40 images, 194 boxes.
- Original split names retained as provenance only: train 132, valid 35, test 8. The source split remains **NOT leakage-safe**: 17 near-duplicate groups, 8 crossing source splits. Must assign complete duplicate groups to a single new split before any training or evaluation.
- Remaining issues: approximate box edges, overlapping visual portions, transparent packaging, food fragments, peel and bone policy. **0 human-verified**.
- Deliverables are the ChatGPT container exports canteen_batch05_40_images_QC.zip, canteen_batch06_40_images_QC.zip and canteen_175_all_images_QC_pending.zip. These ZIP contents are not committed to this branch.
- Main unchanged. No training, no frozen research result modifications.
