# Full V2 annotation QA: 3,096 images (October 13, 2026)

## Exact scope, no inflated verification claim

This report examines the **actual final V2 dataset** exported from the GitHub
Actions reviewed build (commit `8bcf0f780d06`), not merely the preprocessed
Roboflow YAML or 38 selected model-prediction images.

- **2,505** train images + **591** validation images = **3,096** photographs.
- Every final V2 image was opened/verified structurally, matched against its
  corresponding **final V2 label file**, and every labeled class and coordinate
  was range-checked. This is **not the same as an independent person certifying
  all objects in all photos**.
- No malformed/out-of-range annotations; no pair of same-class ground-truth
  boxes with IoU >= 0.90 found by the structural scan.
- Six images with no target boxes: five paper bags/sheets outside the current
  class ontology and one uncapped drink cup (not a plastic drink bottle).
  These are not automatically missing annotations.
- Three boxes have one side under three pixels. Their source photos show
  distant tiny plates/food; flag for downstream small-object review.

### Final V2 object counts at audit export

| Class | Boxes |
| --- | ---: |
| food | 3,678 |
| plate | 881 |
| paper_boat | **0** |
| utensil | 316 |
| hinged_container | 121 |
| napkin | 119 |
| drink_carton | 1,299 |
| condiment_packet | 44 |
| wrapper | 201 |
| bottle | 398 |
| bowl | 131 |

Counts reflect the original 3,096-image reviewed V2 export and do not include
the optional 16 provisional paper-boat / Commons images.

### Actual visual correction: 20 overpadded wrapper boxes

A candidate foreground screen found many `dry_wet_waste` source boxes that
enclose mostly pale background. **20 different original wrapper photographs**
were examined visually with before/after bounding boxes. Their 20 box
replacements were checksum-locked to the original raw labels and committed to
`data/audits/v2_review_2026-10-12/reviewed_overrides.json` (commit
`585f2240e`). The 64 pre-existing replacements are retained (now **84**).

For these 20 corrections only:
- Mean original labeled area = **68.4% of photo**.
- Mean corrected labeled area = **32.0% of photo**.
- Retained the original class ID 8 (`wrapper`).
- Original V1 data and train/validation assignment were **not altered**.
- GitHub Actions V2 full build, label-integrity checks and CPU ML integration
  passed after the patch.

Other white-background bottle boxes are also often too loose, **but a simple
color mask detects colored bottle labels and can truncate clear plastic**.
Never blindly copy those machine-generated bounds into verified ground truth.

### Systematic missing-class risk requiring further visual labeling

Per-source final label counts reveal gaps in many multi-object photos:

- `plate`: 344 source photos, with only **2 photos labeled for utensil**;
  direct visual examples show forks that lack utensil labels.
- `school_lunch`: 467 photos, only **8** with plate boxes and **1** with
  utensil boxes.
- `food_american`: 380 photos labeled food, but only **6** with plate labels.
- `condiments`: 44 visually similar isolated sachet photos; valid-looking
  boxes are **not** 44 independent cafeteria contexts.
- `dry_wet_waste`: highly repetitive white-background isolated trash; the
  wrapper-box corrections are real, but additional bottle and napkin reviews
  are needed.
- No paper-boat annotations in the original final V2 corpus, and validation
  lacks paper boat / napkin / condiment packet positives. Bowl validation
  contains only two positives.

These source-count patterns are **triage evidence, not proof that every
unlabeled target in a source image is actually present**.

### Cross-check of all photos by an independent COCO detector

`scripts/audit_coco_v2.py` runs pretrained **COCO YOLO11n**, separate from the
user-trained cafeteria checkpoint, across **all 3,096 image/label pairs**.
It detects mismatches involving bottle, fork/knife/spoon, bowl and a subset
of food items. Results are written per image to `per_image.csv` and potential
missing boxes to `candidates.csv`. These are **uncertified review leads**;
COCO has no plate, wrapper, napkin, paper-boat, condiment-packet or drink-carton
class. Do not assume a zero-disagreement COCO row means its full 11-class
annotation has been verified.

### Honesty / acceptance condition

**Full all-11-class semantic sign-off of 3,096 photos is not complete.**
It requires an original-resolution review of every photograph and every
object instance, correction of missed and mislabeled objects, a reviewer/
timestamp record for each accepted label file, and re-running validation.
Neither clean file syntax nor a pretrained detector can substitute for this
process. No old mAP results should be represented as metrics for the newly
corrected annotations.

**Do not mark 3,096/3,096 fully reviewed in documentation or college research
descriptions until all individual images have actually passed that step.**
