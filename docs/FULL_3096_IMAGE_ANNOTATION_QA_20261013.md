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

## Updated manual annotations — 2026-10-13 later pass

The original **64** V2 full-image label replacements were followed by **62**
additional individually visualized image-level replacements, for **126** total
unique checksum-locked V2 overrides. No raw V1 label file was edited.

Recent incremental correction groups:
- **20** `dry_wet_waste` snack-wrapper photos: shrink oversized class-8
  boxes on white backgrounds without cropping the actual packaging.
- **12** `dry_wet_waste` clear drink-bottle photos: tighten class-9 boxes
  to the transparent plastic outline rather than label color.
- **30** `plate` source photos: add food, eating utensils, condiment
  sachets, napkins and wrappers that were visibly present but not labeled.
  This includes three initial plate scenes and 27 following corrections.

The later 27 plate-photo corrections add **44** previously omitted boxes.
The 12 bottle fixes are replacements of existing boxes, not fabricated new
bottles. All proposals were checked against image/label overlays and validated
for normalized YOLO box geometry.

GitHub CI checks V2 application and exact V1 label SHA-256 match on every
override. A mismatch or out-of-frame box makes the build fail; both failure
types were found and corrected during this review. Fresh CI results, not
historical baseline mAP, are the appropriate evidence of safe application.

**Full semantic review of all 3,096 images is still NOT finished.**
Mechanical screening of all image/label pairs is complete; most photos
still lack exhaustive independent all-11-class visual certification. The
paper-boat class and missing validation positives remain open data issues.

## Continued original-resolution review — latest batch

**14 additional original-resolution image reviews** have been committed since
the 170-override checkpoint, yielding **184 unique checksum-locked full-image
V2 replacements**, and **34 additional object boxes**. The first 14 are
distributed across V2 train/validation without moving or copying source photos.

Breakdown:
- 4 photos: one drink bottle and food portions on restaurant paper plates
  (10 new boxes). Commit `b35391fdb`, repaired source split name in
  `99a008194` so V2 validation placement is unaffected.
- 3 photos: cake, drink carton, small capped bottle, utensils and food on
  plates (9 new boxes). Commit `4ad0c3e58`.
- 5 photos: folded pizza/calzone, hotdogs, french fries, birthday cake and
  custard pastry on paper plates (10 new food boxes).
  Commit `a9581a465`.
- 2 photos: pizza, edible residual cake/frosting, two black plastic utensils
  and a metal fork (5 new boxes). Commit `de41f20dd`.

All proposed boxes were drawn against the individual photo at original pixel
dimensions; each existing plate/source label was retained; each replacement
stores the original V1 normalized-label SHA-256. No new validation images were
introduced; existing validation annotations changed, so **old mAP figures are
not measurements of this corrected validation set**.

**Coverage is still incomplete:** 184 replacements do not constitute 3,096
independent full-image semantic reviews, and additional images may have omitted
objects. The check-in count is auditable, not a claim of complete certification.

## Continued photo-by-photo review — another 12 images

The next unreviewed review queue was generated from the **actual 3,096-image
export**, excluding previously approved plate records. Out of 61 remaining
plate-source photos with only plate annotations, the first **five** clear
cases were inspected at original resolution and manually augmented:

- `lvis_000000042683`: grilled sausage and baked bun on a paper plate (2 food boxes)
- `lvis_000000226988`: three barbecue portions on overlapping paper plates (3 food boxes)
- `lvis_000000201426`: pizza slice and partly cropped bread on separate plates (2 food boxes)
- `lvis_000000317147`: edible leftover beside cat, on plate (1 food box)
- `lvis_000000072646`: birthday cake and nearby metal eating utensil on an infant's tray (1 food, 1 utensil)

An additional **seven** white-background source images with isolated *clear
plastic* bottles were inspected at original resolution:
`plastic409`, `plastic331`, `plastic234`, `plastic200`,
`plastic379`, `plastic358`, and `plastic202`. Their overpadded full-bottle
boxes were manually tightened to include transparent walls and caps.
A candidate `plastic458` was explicitly **not changed**: its transparent
right-hand portion could not be bounded confidently.

Across these 12 images, **ten boxes were added** (nine food, one utensil)
and **seven existing bottle boxes were replaced**. Current override total:
**196** unique full-image V2 label replacements (previously 184).
The original V1 images and labels remain unchanged and the frozen validation
membership is preserved. The validation *labels* do change where reviewed,
so earlier AP results are not current validation scores.

The source-hash gate identified one transcription error for `lvis_000000042683`
and the exact original-label SHA-256 was corrected in `3fa9a393` before
subsequent corrections were committed. Original-resolution proof images
were saved to `annotations_reviewed_next_12_images.zip` in this
conversation (not added to the private repository).

**No exhaustive 3,096-image visual sign-off has been made.** The 61 plate-only
queue is a prioritization slice, not proof that every photo contains
unlabeled food. Checksums/label geometry and any auxiliary detector are not
a substitute for inspecting target instances in context.

### Additional bottle review: four source photographs

Four more plain-background photos were examined individually with old (cyan)
and revised (red) boxes: `plastic381`, `plastic167`, `plastic277`,
and `plastic104`. These include bottles clipped by the image border;
only the visible bottle and cap are boxed, with no automatic color-only
masking. Their original label hashes are recorded in
`reviewed_overrides.json`, bringing the V2 count to **200 unique
checksum-locked image-level replacements**. In the current turn, **16
additional images** received review-backed changes: 5 plate scenes
(10 newly added food/fork boxes) and 11 isolated bottle scenes
(11 full-bottle box replacements). The review evidence is available as
`annotation_review_16_new_images.zip` in the conversation.

The 3,096-image structural audit is complete; all-11-class exhaustive visual
verification is still not complete. Old validation scores are not valid for
this corrected set without reevaluation.

## Ongoing October 13 verification: corrected plate-only labels

Six additional previously unreviewed plate images were visually checked at
original photo resolution, with **13 missing boxes** added (food and eating
utensils): LVIS `144065`, `167520`, `212248`, `351608`,
`371950`, `478407`. Four more original-resolution reviews
added **9 boxes**: LVIS `143143`, `384981`, `479350`, `499705`.
They preserve every pre-existing plate box, source checksums, image
split membership, and all raw V1 labels. The total V2 reviewed override count
is now **210**. These images are a subset of the full unreviewed corpus,
not a claim that all 3,096 annotations have been signed off.

A visually tempting candidate `lvis_196888` was *not* annotated as food:
the dark object on the plate is a bird. The crowded `lvis_506011`
remains flagged for further human review because the tabletop contains
ambiguous small decorations and multiple eating objects. The original
photos were checked to avoid using a pretrained detection as ground truth.

**Additional semantic cross-check requested:** a pretrained COCO-only object
detector scan of all 3,096 final V2 images with separate exported candidate
queue. Its proposals must remain *unverified*, and its COCO ontology
cannot resolve plates, wrappers, napkins or packets.

## Native-resolution source review and durable 3,096-row ledger (October 13, 2026)

A new per-image status ledger is stored at `data/audits/v2_review_2026-10-12/full_11_class_review_ledger.csv`.
It includes exactly **3,096 source image keys** and distinguishes accepted
full-image reviews from thumbnail screening, prior partial reviews and
pending cases. The ledger is rebuilt from the exact existing review records,
not by pretending a contact sheet constitutes 11-class verification.

At this checkpoint **241 / 3,096** image records have an accepted
complete visual-review record: **239** checksum-locked
complete label overrides and **2** directly inspected,
unchanged full-image annotations. **2855** images are not
fully signed off, including explicitly flagged follow-ups.

This native-resolution batch inspected four new `food_american` photos:
`test__images_jpg.rf.8725f085ad0ededab7494a40cdf72491` (add missing fries),
`train__3-16_png.rf.dac88c9539225132ee07b229eeb92204` (correct
central burger and add two edge-clipped burgers), plus two unchanged
single-food photographs `f08c83f04657` and `f3c0e618c95`.
The two corrected images each retain exact source-label SHA-256 gates;
source image hashes and reviewer notes are stored in the ledger.
Two additional food photographs (`2e1375c3`, `8a509a19`) remain
explicitly pending because grouping/plate boundaries are ambiguous.

Previous transient GitHub Actions failures were investigated: one earlier
run failed on an exact source SHA-256 typo for `plastic189` and was repaired
in the subsequent successful run; integrity checks remain enabled.
Neither new label edits nor ledger changes imply improved model accuracy.

## Original-resolution all-11-class audit checkpoint — 2026-10-13

The durable 3,096-row status ledger now records **258 fully reviewed photos**:
**240** corrected V2-only label sets with exact source-label SHA-256 locks and
**18** individually inspected photos whose complete labels required no change.
**2,838** photos remain without full 11-class sign-off. A further **17**
food_american photos were examined at native resolution in this pass:
12 were fully verified (11 unchanged and one corrected with fries and plate
boxes); five more simple photographs were subsequently verified unchanged.
Additional ambiguous food, utensil, serving-plate, pan and parchment scenes
were explicitly left `pending_followup` in the ledger rather than counted.
The final 3,096-image structural scan is not a semantic certification.

The new corrected photo is `test__images_jpg.rf.2e1375c3d05007414011f60f20dd3797.jpg`.
Its original 201x251 pixels and proposed overlay were inspected. The burger
box was retained and separate food/fries and plate-rim boxes were added.
The source-label checksum gate passed the full reviewed build. Earlier
model accuracy and AP figures are not reevaluations of this corrected data.
