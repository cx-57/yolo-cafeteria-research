# Canteen Batch 03 — visual QC checkpoint

**Status:** PENDING independent annotation QC. No modifications to main.

Sources: Roboflow original exported v1 COCO segmentation dataset, 175 photographs. The current checkpoint examined 40 additional photographs individually at the stored full-resolution dimensions. Prior batches covered 15, so total individually inspected is **55/175**; **120 remain uninspected**. All AI authored, **0 human verified**.

Batch 03 includes 40 corrected, re-grouped YOLO annotation files in the **local ZIP export**, matching 40 unchanged original source images, source-resolution bounding-box JPEG previews, original metadata, per-box audit, and change log.

Seven classes: 0 food; 1 plate (including rigid serving bowls); 2 utensil; 3 hinged_container; 4 drink_carton; 5 wrapper; 6 bottle.

Full 175-image **provisional** archive also generated: every photograph has a .txt label file and visible bounding-box preview, but 120 images retain uninspected prior converted proposals. They must not be considered complete/approved. Known issues include duplicate food portions, missed bowls/utensils/wrappers, meat-on-bone and banana peel policy uncertainty, name collisions and near-duplicate split leakage.

Numerical validation: 175 JPEG hashes preserved, 175 corresponding YOLO label files, 175 annotated previews at source pixel dimensions, 959 total provisional boxes across the full archive with classes within [0,6] and coordinates in [0,1]. Counts: food=520, plate=188, utensil=247, wrapper=3, bottle=1, hinged_container=0, drink_carton=0.

No training or frozen research changes. Further individual review is required before declaring the full dataset verified; the midpoint QC checkpoint has not been reached. The archive ZIPs are generated in the chat container, not hosted as GitHub Actions artifacts.