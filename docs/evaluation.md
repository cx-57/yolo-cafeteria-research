# Seven-class evaluation

The default evaluation now uses the frozen **186-photo repaired validation cohort**
at external `artifacts/validation-label-audit-20261030/audited-eligible.json`.
The current release scores **85.69% mAP50 / 64.90% mAP50–95** on it.
Per-class measurements are in `results/baseline.json`. Labels were reviewed without
model predictions, with uncertainty notes retained; they are AI-assisted and not
human certified. Only two hinged-container targets remain. This is development
validation with historical checkpoint exposure, not an independent school test.

The prior 61% headline used incomplete annotations. On the same 186 photos with
unchanged predictions, annotation repair changes 61.39% / 48.52% to 84.68% / 64.22%.
On the full unchanged 248-photo membership, corrected labels score 83.64% / 64.43%.
Those changes correct measurement; they are not model-training gains. The 62 quality
exclusions were made by blind visual policy review before rescoring.

After annotation repair, a fixed three-epoch classification-only utensil trial
improved utensil from 68.37% / 42.30% to **71.98% / 44.43%**, and the full primary
pipeline from 84.68% / 64.22% to **85.20% / 64.53%**. Only that checkpoint was
activated. The other three remain unchanged. All non-classification parameters and
BatchNorm running statistics were verified unchanged at checkpoint precision.
Fresh native four-model inference confirmed the composed result on both cohorts.
That 640-pixel pipeline scored **90.28% / 67.08%** on the historical 124-photo cohort.
Backups, training recipes, resumption provenance and native activation receipts
are external under `artifacts/classifier-retention-20261031/`. No test inference
was used. These gains do not establish independent school generalization.

A fixed three-epoch box-regression-only follow-up at learning rate 0.000002
scored 71.95% / 44.09% for utensil, versus the active 71.98% / 44.43% on the same
repaired cohort. It was rejected; active weights, labels and default scores are
unchanged. Source hashes, frozen-classifier checks and the comparison are external
under `artifacts/utensil-localization-20261031/`. No test images were evaluated.

The subsequent weights-only feature/classifier trials copied all training labels
byte-for-byte and left the repaired validation manifest fixed. Plate/wrapper
mAP50 rose slightly, but mAP50–95 fell; bottle declined on both metrics. Both
updates were rejected. Their artifacts are under `artifacts/feature-adaptation-20261031/`.

One uniform 800-pixel input trial improved utensil and bottle, but damaged other
classes. A two-model resolution profile also failed a historical bottle check and
was rejected. The accepted profile uses **800 pixels only for utensil**, with all
other models at 640. Fresh native inference gives utensil **74.77% / 46.09%** and
the primary pipeline **85.59% / 64.76%**, versus 71.98% / 44.43% and 85.20% / 64.53%
before this input change. Every other primary class metric is unchanged.

Weights, annotation bytes, validation membership, confidence/NMS settings and AP
matching are unchanged. Receipts explicitly record each model's input settings;
per-model overrides can change resolution only. This is a validation-selected
inference-profile gain with higher compute cost, not a weight-training gain.
The historical 124-photo reference changes slightly to **90.24% / 66.86%**.
Backups and native receipts are under `artifacts/resolution800-20261031/`.
No test data was used, and these remain provisional development measurements.

The following table is the **pre-classifier-update historical 124-photo benchmark**, preserved
separately at external `artifacts/simple-specialists/shared-seven-active.json`.
Four frozen models ran on those same 124 development images, with reviewed
coverage for the seven retained classes. Packet labels are excluded at the user's
request; no images or other class labels were removed. Class IDs 0–6 are unchanged.
Ordinary single-pass 640, NMS IoU 0.7, batch 1; no ensemble, verifier or TTA.
AP uses Ultralytics matching at IoU 0.50:0.05:0.95 and standard integration.

| Class | AP50 | AP50–95 | Objects |
|---|---:|---:|---:|
| Food |96.85%|68.31%|81|
| Plate |76.40%|62.00%|49|
| Utensil |81.03%|54.30%|65|
| Hinged container |99.50%|66.41%|12|
| Drink carton |94.15%|70.15%|29|
| Wrapper |89.56%|67.53%|21|
| Bottle |87.21%|74.34%|11|
| Macro mean |89.24%|66.15%|268|

The earlier seven-class baseline was 88.43% / 63.69%. The current change is an
eight-epoch food/container/carton detection-head adaptation, not a gain caused by
dropping condiment packet. The other three checkpoints and all inference settings
are unchanged. Both development and school validation participated in acceptance.

Labels remain provisional and AI-assisted. Known cross-model training overlaps
were excluded, and incomplete annotations were held. Exact bytes, decoded pixels
and available group/capture identities were checked. Capture-session independence
and absence of COCO pretraining overlap are not certified. Historical validation
was repeatedly used for selection, and some classes have few objects. An
independently collected cafeteria test with human-certified labels is still needed.

The seven-class manifest records its parent eight-class manifest hash. Full source
reviews, historical prediction receipts and packet experiments remain in the
external data/research store. Frozen predictions with an old eight-class recipe
cannot be silently reused as a new four-model receipt.

Layout references: [Ultralytics](https://github.com/ultralytics/ultralytics) and
[YOLOv5](https://github.com/ultralytics/yolov5).

## Historical full audited school cohort

All 123 school photos received an individual AI visual label audit. Revised labels
live separately from the preserved draft. This earlier cohort excluded 17
unresolved photos and retained the original day-based splits: 30 train, 29 monitor,
47 test. Do not train on monitor or test days. Labels are not human certified, and
the test day had already been inspected before this audit.

The four unchanged checkpoints' existing predictions were rescored without new
inference or training. On the 47 eligible test-day photos:

| Class | AP50 | AP50–95 | Objects |
|---|---:|---:|---:|
| Food |53.98%|33.32%|43|
| Utensil |90.85%|82.58%|12|
| Drink carton |0.00%|0.00%|9|
| Supported three-class macro |48.28%|38.63%|64|

Plate, hinged container, wrapper and bottle lack test-day ground truth; no AP is
reported for them. This three-class macro is not comparable to the seven-class
public development macro above.

Food's original 50.92% / 31.92% score used 53 photos. The higher eligible-cohort
score results from excluding unresolved images: old and audited labels give
identical scores on the same 47 photos. Across all 106 eligible school photos,
food's label-only change is 47.29% / 29.72% to 48.75% / 30.96%. This is an annotation
correction, not a model accuracy gain.

External-store locations:

- Historical dataset:
  `data/school-submission-20261027/label-audit-20261028/dataset-before-quality-filter/data.yaml`.
- Audit: `data/school-submission-20261027/label-audit-20261028/`.
- Manifests, frozen receipts, cohort comparisons and CLI confirmation:
  `artifacts/school-audited-evaluation-20261028/`.

## Historical 40-photo clear, in-scope school cohort

The user subsequently excluded trash-can scenes, blurred photos or retained
targets, unresolved labels, and food outside the documented broad training families.
Every photo has an inclusion/exclusion record. At the user's request, food matching
was broadened further: pasta covers tortellini, potatoes cover tater tots, and mixed
rice/meat/vegetable meals are not rejected for different ingredients. Hummus is
included as an inferred prepared plant-food family match; this does not prove that
the exact dish or chickpeas occurred in training. Original source categories include
mixed rice, meat/vegetable meals, salads and vegetable-based cooked food.

This earlier school cohort contained **40 photos: 19 train, 9 monitor, 12 test**.
Original day splits and all labels on included photos are unchanged. No trash or
blurred photo was deleted; excluded photos remain available in the original data.
The historical 106- and 33-photo datasets are archived separately. Seven clear
training/monitor photos were restored under broader food-family matching; blur,
trash and unresolved-label exclusions were not relaxed. No test photos were added.

The clean test subset has **seven food instances in five related nugget photos and
seven negative photos**. After adaptation, food AP50 was **64.19%**, AP50–95 **24.93%**, versus
**43.31% / 16.93%** before adaptation on identical images and labels. No other class
has test ground truth. This small, correlated subset is not a seven-class benchmark
or independent generalization evidence. Test photos were already inspected and
participated in release acceptance. The nine-photo monitor
has only three food, one utensil and one carton instance, so it is inadequate for
robust tuning. Only training and monitor days influenced fitting/guard selection;
the inspected test day was used once for candidate release comparison, then checked
again with the unchanged candidate through the ordinary pipeline for reproducibility.

### Detection-head adaptation and remaining errors

Training used 19 clear school train photos (four exposures each) plus 171 original
train photos passing a visual quality/annotation-completeness screen. Of 223 replay
candidates, 52 were rejected for trash scenes, blur or problematic labels. No fitting
image matched public validation or non-train school photos by bytes/decoded pixels.
Only YOLO11 Detect layer 23 was trainable; backbone, neck and BatchNorm statistics
were fixed. AdamW lr0=0.0001, lrf=0.2, batch=1, nbs=8, imgsz=640, seed=42, eight
epochs, final checkpoint only; no epoch sweep. Mosaic=0.25, closed for the last two
epochs; scale=0.25, translate=0.1, mixup=0, fliplr=0.5. Inference stays single-pass
640, native NMS=0.7, confidence floor=0.001. Scores were not artificially inflated.

Monitor food improved from 11.17% / 2.43% to 35.00% / 9.13%. Each affected public
class had to stay within two percentage points on both AP metrics; all improved.
The one candidate also had to improve both metrics on the already-inspected school
test before activation. Stronger extra food NMS=0.45 was tested on monitor/public
first and rejected; it is not in the active pipeline. No changes followed test results.

At a fixed diagnostic confidence cutoff of 0.25, food TP/FP/FN changed from 1/3/6
to 3/2/4 on the 12 test photos. At 0.10, the tradeoff was 6/12/1 to 5/8/2. These
cutoffs are diagnostic examples, not newly selected operating settings. Duplicate
and oversized boxes, weak cropped-food confidence, and inconsistent food-unit
annotations in the original sources remain issues. The system is improved, not
fully solved; current labels remain AI-assisted and no overfitting-free claim is made.

### Rejected serving-label follow-up

A subsequent audit reviewed 80 replay food photos. Six ambiguous/incorrect source
photos were excluded, and 50 admitted images received serving-unit or boundary
corrections in separate label exports. On the same 165 retained replay images,
food boxes changed from 238 to 192; original labels and all school labels were
preserved. A separate audit of 22 public food-positive photos held two images with
opaque sealed cups labeled as visible food, producing a secondary 122-photo guard.
The original 124-photo reference benchmark was retained unchanged.

One further fixed eight-epoch head-only trial from the active model used these
165 replay images plus the same 19 school train photos. It was rejected: monitor
food AP50 fell from 35.00% to 33.50%, while AP50–95 rose from 9.13% to 10.05%.
Original public food changed from 96.85% / 68.31% to 94.57% / 69.16%; it failed
the predefined retention guards. Candidate checkpoints were deleted and the school
test was not rerun. The active 64.19% / 24.93% model remains unchanged.

The same fixed extra food NMS=0.45 was also checked on the active adapted model:
monitor food changed from 35.00% / 9.13% to 36.14% / 4.89%, so it was rejected
without school test scoring. These failures are retained rather than tuning on
the seven school test instances. That monitor had only three food targets;
additional independently collected clear food sessions are needed for stronger
validation. Versioned source-label corrections, audit evidence, protocols and
comparisons are external under `artifacts/food-serving-alignment-20261028/`.

## Current uncropped-food school cohort

The user subsequently excluded any school photo where food is cut off by the
frame and confirmed **one box per serving**, not one per chicken piece. A chicken
pile sharing a serving is one food target; a genuinely separate serving/location
remains separate. No model rule limits an image to one food detection: the existing
pipeline retains multiple boxes with max_det=300 per detector. Serving-unit labels
are versioned separately; current model errors are not hidden by a new merging rule.

All 20 food-positive photos in the prior active cohort received a whole-frame
visual edge review. Nineteen contain cut-off food and were excluded as whole photos
to avoid turning the remaining visible food into unlabeled negatives. No pixels,
individual food labels or original files were deleted. Only IMG_3405 has fully
in-frame food. The dataset now has **21 photos: 8 train, 6 monitor, 7 test** with
original day splits and all previous quality exclusions preserved.

The one food-positive photo is in the original training split and contains two
distinct servings. Neither monitor nor test contains food ground truth, so food
AP50/AP50–95 are **undefined**, not zero and not the historical 64.19% / 24.93%.
The historical 40-photo cohort, receipts and scores remain archived. No new model
was fitted on this one-food-photo dataset, and current checkpoint hashes are
unchanged. The seven negative test photos still produce two food boxes at a fixed
0.10 confidence cutoff and one at 0.25; these are false-positive diagnostics only,
not a substitute for detection accuracy. More complete-serving photos from separate
sessions are needed to restore a meaningful food validation set.

Current external-store locations:

- Dataset: `data/school-submission-20261027/active-seven-class/data.yaml`.
- Current crop decisions and user-confirmed serving policy:
  `data/school-submission-20261027/uncropped-food-scope-20261028/`.
- Original food-family source evidence: `data/school-submission-20261027/clean-scope-20261028/training-food-coverage.json`.
- Current filtered manifests, receipts and undefined-AP diagnostics:
  `artifacts/school-uncropped-evaluation-20261028/`.
- Unchanged model recipe, historical manifests, native receipts, before/after comparison and
  original checkpoint/config backups: `artifacts/food-detection-fix-20261028/`.
- Pre-adaptation receipts: `artifacts/school-broad-food-evaluation-20261028/`.

Do not request food AP from the current target-negative school test manifest;
the evaluator rejects missing ground-truth support. Default evaluation now uses
the repaired 186-photo cohort; the original 124-photo cohort is a separate retention check.

On 2026-10-31 the user permitted sharp cropped food for training only. Nine
reviewed cafeteria photos were admitted and two soft candidates held, bringing
the training pool to 16 school originals and 300 balanced school views. Validation
and test membership, labels and bytes stayed frozen. The new combined dataset is
`data/combined-seven-cafeteria-crops-20261031/data.yaml` in the external store.
Its explicit `allow_cropped_food_train` setting never relaxes held-out crop checks
or permits boxes outside the image. Exact quarter turns preserve box geometry;
they simulate phone rotation, not additional camera viewpoints.

A fixed three-epoch continuation used all 300 school views per trained specialist,
with reviewed public replay, training classification branches only. Food/container/
carton, plate/wrapper and utensil were trained; bottle stayed unchanged because
the school training photos contain no bottle positives. The resulting seven-class
pipeline scored **85.31% mAP50 / 64.57% mAP50–95**, below the retained
**85.59% / 64.76%** on the same 186 photos. No candidate was promoted. Checkpoint,
source-annotation and input hashes were verified. The test split was not scored.
Results and frozen-weight checks are in
`artifacts/cafeteria-full300-push-20261031/report.json` and its specialist folders.

At the user's explicit request, the retained pipeline was then scored once on the
frozen 248-photo test cohort: **67.00% mAP50 / 54.10% mAP50–95**. Per-class AP50
was food 96.10%, plate 27.72%, utensil 64.44%, container 99.50% (four objects),
carton 97.76%, wrapper 14.11% (18 objects), and bottle 69.39%. This is the raw-label
baseline, before the separate test annotation repair described below.
The six school test photos contain only one labeled carton; carton AP was zero,
and school seven-class/food AP is undefined. No current train/validation group,
byte or pixel overlap was found, but historical inherited-checkpoint source
exposure cannot be ruled out, so this is not certified as an independent test.
Models, scoring settings, labels and cohort membership were unchanged. These
results are now exposed and must not be treated as an unseen model-selection set.
The original raw-label summary, complete frozen manifest,
prediction receipt and preflight checks remain in the external
`artifacts/frozen-test-20261031/` folder.

The user then requested the same label-review rules used for validation. Three
isolated reviewers inspected all 248 photos across all seven classes, using
original pixels and preexisting source annotation proposals without predictions
or model scores. Each uncertain case received a second reviewer; root checked
samples, quality flags and difficult corrections. Root already knew aggregate
raw-test scores, but did not inspect per-image detector outputs. Clear omissions,
scope errors and bounds were corrected. There are 103 substantive changed
annotation sets (109 serialized row sets including ordering/format differences),
with explicit uncertainty retained on 47 photos. These labels remain provisional
AI-assisted annotations, not independent human-certified ground truth.

Annotations, adjudications and all quality decisions were frozen before rescoring
the identical saved detections. All **248 test photos** remain in the primary
repaired cohort, scoring **87.61% mAP50 / 67.23% mAP50–95**. Per-class AP50 is food
83.91%, plate 86.88%, utensil 90.65%, container 99.50% (four objects), carton
97.55%, wrapper 84.63%, and bottle 70.13%. Added food targets include missed
servings in crowded scenes; repairing labels does not guarantee a higher score
for every class. The six school photos still have one identifiable carton target
and no measurable seven-class or food-only school AP.

Applying the existing blur/bin/cropped-food rules creates a separately labeled
**158-photo quality-eligible subset**, scoring **92.31% / 69.98%**. The same158
photos with original labels scored **71.02% / 57.20%**. This separates annotation
correction from cohort filtering; the 92.31% result is not the full test score.
No models, inference/scoring settings, original files, or primary membership
changed. Both repaired scores were reproduced with the standard evaluator using
frozen prediction receipts. These are post-test measurement corrections, not
training gains or a new independent test. Historical checkpoint source exposure
still cannot be ruled out.

`results/test.json` now records the repaired primary result, raw-label baseline,
secondary quality cohort and limitations. External
`artifacts/test-label-audit-20261031/` contains all review evidence, source hashes,
`audited-all248.json`, `audited-eligible.json`, `frozen-audit.json`, `rescoring.json`,
and CLI confirmations. Original combined-dataset test labels are preserved;
use these audited manifests explicitly for repaired test evaluation.

The final release-training round continued all four specialists for two fixed epochs
from their retained checkpoints, using all 300 school views from 16 independent
training parents plus reviewed public replay. Per-specialist training views were
405 (food/container/carton), 627 (plate/wrapper), 421 (utensil), and 592 (bottle).
Only feature block 22 trained; every other parameter and all BatchNorm statistics
stayed fixed. Learning rate was 0.000002 with nbs 32, batch 1, no online geometric
warps, and unchanged inference settings. Training losses in `results/release.json`
are fit diagnostics, not held-out accuracy.

Validation selected only plate/wrapper and utensil. Wrapper AP50 changed from
76.98% to 77.47%, and utensil from 74.77% to 74.94%; utensil AP50–95 changed from
46.09% to 46.77%. The primary pipeline changed from 85.59% / 64.76% to
**85.69% /64.90%**. Food/container/carton and bottle updates
were rejected. Fresh native four-model inference confirmed the primary 186 result.

The selected pipeline then scored **87.05% mAP50 /66.84% mAP50–95**
on the same frozen repaired 248 test photos. Its separate quality-eligible 158
result is **92.08% /69.84%**; historical 124 retention is
**90.46% /66.91%**. Labels, photo membership, confidence/NMS
settings and scoring stayed unchanged. Test results gated the user-requested
GitHub publication at 85% AP50 on both primary validation and full repaired test;
they did not select checkpoints or tune training. The test benchmark was already
exposed, so this does not establish independent generalization.

Canonical weights/configuration were backed up before activation. The release
worker verifies hashes and native receipts and restores old files on a failed
check; its late-write rollback path was exercised using temporary fake files
and mocked inference without GPU/model loads. Source review found no outstanding
critical/important issues; the code suite passed 29 tests. Training, release gate,
receipts, backups and rollback-test evidence remain external in
`artifacts/release-training-20261031/`. Compact release metrics and checkpoint
hashes are in `results/release.json`.
