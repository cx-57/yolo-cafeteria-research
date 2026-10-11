# New public source dataset screening (2026-10-10)

## Input audits
- TartanHacks v2: 1568 images. Source labels include 180 napkin images (182 objects), 236 utensil images and 212 rigid-plastic-container images.
- Food Packages v2: 1808 images. 140 photos have three-compartment boxes, 70 have two-compartment boxes, but there are no verified NHS-like clear four-cell GS6-4 containers.
- Every image in both archives was decoded/verified with Pillow. All selected YOLO boxes were checked for normalized bounds.

## Curation (conservative, scene-aware)
- Selected **35** napkin photos across **7 distinct recorded clips**, based on visually reviewed class contact sheets, temporal thinning, and source box validity.
- Selected **27** additional white disposable utensil photos across **9 distinct clips** (YOLO class 3).
- Rejected the TartanHacks rigid_plastic_container category (opaque takeout packages, not clear four-cell NHS tray).
- Rejected Food Packages for `compartment_container` (project class 4); its beige 2/3-cell meal trays do not match the school product. No files from Food Packages promoted.
- Other classes (cups, straws, small containers, plastic bags, lids) were not silently mapped into unrelated project classes.

## Critical split policy
Roboflow's original train/valid/test partitions have **different frames from the SAME videos in different splits**, so using those original splits would leak scenes. Each imported manifest row records `capture_group`. All frames sharing a capture_group MUST remain in the same downstream split. Reserve separate NHS scenes for actual test evaluation.

## Limitations
The selected images were reviewed with class contact sheets, not individually full-resolution annotated in all 11 target classes. They are *provisional* until complete multi-class box review. Original file sha256, source path, split, license, and capture group are saved to the manifest. Unselected source images are excluded from GitHub.
