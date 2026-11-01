# Cafeteria object detection

Four YOLO specialists detect seven classes:

| Model | Classes |
|---|---|
| Food/container/carton | Food, hinged container, drink carton |
| Plate/wrapper | Plate, wrapper |
| Utensil | Fork, knife and spoon union |
| Bottle | Disposable plastic drink bottles |

```sh
./scripts/run_specialists.sh predict /path/to/image.jpg --device mps --output runs/predictions.json
./scripts/run_specialists.sh evaluate --device mps --output runs/evaluation.json
```

Prediction accepts an image or directory. Use `--device cpu` for CPU inference.
Output is normalized `[x1,y1,x2,y2,confidence,class_id]`. IDs 0–6 retain their
existing meanings: food, plate, utensil, hinged_container, drink_carton, wrapper,
bottle. Utensil uses an 800-pixel input; the other three models use 640.
All use single-pass inference, NMS IoU 0.7 and batch 1, loading sequentially.
The 0.001 confidence floor supports AP; filter scores for display.
Only input resolution differs: annotation policy, validation cohort and AP scoring
remain common to all seven classes. Receipts record each model's applied settings.

## Setup and assets

Requires Python 3.12 and `pip install -r requirements.txt`. The launcher uses the
existing external environment on this Mac; set `CAFETERIA_PYTHON` to override it.
Place the four pinned checkpoints in `weights/` using the filenames and hashes in
[configs/specialists.json](configs/specialists.json). Weights are supplied separately
and excluded from Git.

Datasets, annotations and historical research live in the sibling
`yolo-cafeteria-research-assets/` folder. Override with `CAFETERIA_STORAGE`.
Runtime caches default to `~/Library/Caches/yolo-cafeteria-runtime/`;
override with `CAFETERIA_RUNTIME`. Prediction only needs the four checkpoints.
Evaluation defaults to external
`artifacts/validation-label-audit-20261030/audited-eligible.json` (186 repaired photos);
a custom manifest can be passed as a positional argument.

## Evaluation

The primary repaired validation set scores **86.36% mAP50 / 65.25% mAP50–95**
with the active specialists. See [per-class results](results/baseline.json).
Labels are AI-reviewed with documented uncertainties, not human certified.

The release models score **87.78% mAP50 / 67.24% mAP50–95** on all
248 repaired test photos. The separate 158-photo subset meeting the existing
blur/crop rules scores **92.08% / 69.84%**; it is a narrower cohort,
not the full test split. See [test results and caveats](results/test.json) and
[training/release checks](results/release.json). These post-test AI-reviewed
measurements retain uncertainties and are not a fresh independent test. The earlier
annotation audit's before/after comparison is preserved in the evaluation notes.
Original combined-dataset test labels remain archived. For repaired test scoring,
pass the external `artifacts/test-label-audit-20261031/audited-all248.json`
manifest to `evaluate`; `audited-eligible.json` is the separate filtered version.

An earlier specialist version scored
**92.25% mAP50 / 71.96% mAP50–95** on a
separate, quality-screened 84-photo subset of the public development cohort, with
all seven classes represented. This is the same model on a narrower benchmark,
not a new accuracy gain or a claim of 90% school-photo performance. The original
124-photo benchmark below remains a separate historical retention check.

Before that update, the seven retained classes scored
**89.24% mAP50 / 66.15% mAP50–95** on the same
124 development images after an eight-epoch food/container/carton detection-head
adaptation. The other three checkpoints were unchanged at that stage. Packet removal itself did
not improve weights. These are provisional AI-assisted development results, not
independent cafeteria test accuracy.
See [evaluation notes](docs/evaluation.md) for that historical cohort.
The current release scores **91.87% / 68.12%** on that historical 124-photo
retention benchmark; the repaired 186-photo set remains primary.

The earlier school-photo audit is preserved separately. At the user's request, that
school dataset excludes trash-can scenes, blurred photos/targets, unresolved labels,
photos with food cut off by the frame, and food outside the broad training families.
Mixed-meal and prepared
plant-food matches are allowed; exact ingredient exposure is not certified. External
`data/school-submission-20261027/active-seven-class/data.yaml` version has **21 photos:
8 train, 6 monitor, 7 test**. Only one training photo contains food; no food-positive
monitor/test photos remain, so **current school food AP is undefined**. Labels use
one box per serving, including a whole chicken/nugget pile, not each piece. The
model supports multiple serving detections; there is no one-box-per-image cap.

The unchanged model previously scored **64.19% / 24.93%** (AP50 / AP50–95) on a
different, cropped-inclusive 12-photo validation set, versus **43.31% / 16.93%** before
adaptation. That historical result is not accuracy on the current filtered data or
independent generalization. Original images, labels, weights and earlier dataset
versions are preserved. No replacement model was deployed after this crop filter.

## Combined public and school split

The initial merged seven-class dataset is preserved externally at
`../yolo-cafeteria-research-assets/data/combined-seven-school-20261030/data.yaml`.
It contains 2,573 images: **2,077 train / 248 validation / 248 test**
(80.7% / 9.6% / 9.6%). All 21 school photos eligible at that stage are included:
8 train, 7 validation and 6 test, with capture days kept together and prior school
training days retained in training. All seven classes have labels in every split.
This new dataset does not change the active models or their existing benchmark.

The build remaps the larger reviewed public corpus to seven classes, overlays
selected later corrections, and excludes known visual-review rejects, annotated
frame-touching food, retired-class-only public images and conflicting duplicates.
Known capture groups, filename variants and close perceptual hashes stay within
one split; sources previously restricted to training remain there. Original
datasets are preserved. Build, exclusion and verification receipts are under
external `artifacts/combined-split-20261030/`.

Public labels remain provisional; this is not a fresh exhaustive annotation audit.
Validation and test each contain only four hinged-container instances. These
photos have historical training/evaluation use: the new split is **not an
independent test of existing checkpoints**. Start new comparisons from generic
pretrained weights and reserve newly collected school days for independent testing.

## Reviewed combined dataset

Use the newer dataset at
`../yolo-cafeteria-research-assets/data/combined-seven-school-oct29-20261031/data.yaml`.
It contains **2,372 training views, 186 validation photos and 248 test photos**.
Training contains 2,052 real photos plus 320 school variations. The school portion
is 340 views from **20 real training photos**, including nine reviewed, sharp
cropped-food photos admitted for training only. Soft targets and bin scenes stay
excluded. Exact quarter turns move boxes with the image; mild lighting variations
add no objects or camera viewpoints. Augmentation never crosses splits.
The dataset explicitly sets `allow_cropped_food_train: true`; the validator still
rejects cropped food in validation/test and boxes outside the image in every split.
The four October 29 originals contribute 100 views total (25 each).
Four sharp photos from the October 29 submission entered training; 17 were held
for blur or annotation uncertainty. The two-epoch food specialist trial was rejected
because food mAP50–95 fell from 65.01% to 64.97%; active models are unchanged.
The whole capture day belongs to training, so its remaining photos are not an
independent test. Public training images are referenced from the previous working
dataset; keep that dataset available.

The prior seven-parent version remains available at
`../yolo-cafeteria-research-assets/data/combined-seven-reviewed-20261031/data.yaml`.
Quality filtering changes the original 80/10/10 proportions; surviving photos keep
their frozen roles rather than being reassigned after model training.

The initial merged corpus omitted later specialist annotation corrections.
Blind image review repaired 126 of the 248 validation annotation sets. The same
unchanged predictions score 83.64% mAP50 / 64.43% mAP50–95 on those same 248
photos after label repair. On the same 186 policy-eligible photos, correcting
labels changes 61.39% / 48.52% to **84.68% / 64.22%**. These are measurement
corrections, **not model-training gains**. The original 124-photo benchmark remains
separate and unchanged.

A subsequent three-epoch classification-only utensil update raises primary
validation to **85.20% / 64.53%**. Utensil itself rises from **68.37% / 42.30%**
to **71.98% / 44.43%**. The backbone, box-regression state and BatchNorm running
statistics were retained at checkpoint precision. Only the utensil checkpoint was
activated after fresh native verification; the other three remain unchanged.
The previous checkpoint/config and all experiment receipts are external under
`artifacts/classifier-retention-20261031/`.

The latest input-resolution change raises utensil to **74.77% / 46.09%** and the
full primary pipeline to **85.59% / 64.76%**, using 800 pixels for utensil and 640
for the other models. Checkpoint bytes, annotations, image membership and scoring
are unchanged; inference compute increases. An 800-pixel bottle profile improved
the primary cohort but regressed substantially on the historical reference, so it
was rejected. All trials and the accepted profile are external under
`artifacts/resolution800-20261031/`.

Reviews and quality decisions were made without model predictions. The 62
validation exclusions follow the requested no-bin, no-noticeably-blurred-target,
and no-frame-cropped-food-serving policies; difficulty alone is not an exclusion.
Training merges prior specialist annotations and the new owner-specific reviews,
while unreviewed public annotations remain provisional. Originals are preserved.
Receipts, uncertainty notes and hashes live externally under
`artifacts/validation-label-audit-20261030/` and
`artifacts/combined-seven-reviewed-20261031/`.

The repaired validation labels are AI-assisted, not human certified, and some
ambiguous cases remain flagged. Only two hinged-container objects remain in the
186-photo cohort. Historical checkpoint exposure is possible, so these figures
are development results, not independent school accuracy. The test split has not
been evaluated in these experiments.

## Orientation-aware candidate training

The `train` command in this section creates a separate seven-class candidate.
Specialist continuation uses the existing specialist checkpoint and its matching
class view, as recorded under external `artifacts/classifier-retention-20261031/`.

`configs/train-angles.yaml` provides native online augmentation: ±15° rotation,
±2° shear, mild perspective, scale/translation and lighting changes. Mosaic,
mixup, cutmix and vertical flips are disabled. Native YOLO transforms the labels
with each image; validation/test do not receive random training augmentation.
Geometric augmentation can create partial views with clipped matching boxes;
source eligibility excludes cut-off food unless the dataset explicitly permits
reviewed training-only crops with `allow_cropped_food_train: true`.

```sh
./scripts/run_specialists.sh preview --data /path/to/data.yaml --output /tmp/angle-preview
./scripts/run_specialists.sh train --data /path/to/data.yaml --model /path/to/local.pt \
  --output /path/to/new-candidate --epochs 20 --device mps
```

Normal training requires the seven-class mapping above, explicit label files,
and ground-truth support for every class in train and validation. It rejects exact
image duplicates across splits, invalid boxes, unapproved cropped-food source
boxes, repairable JPEGs,
and profile overrides that alter class scope, resume or output paths. Audited
inputs are copied into the new run to freeze membership and protect originals
from native cache/repair writes. Checkpoints must be local; no automatic downloads.
`--smoke-test` forces one functional epoch and is not an accuracy experiment.

The augmentation workflow is experimental: the completed 20-epoch single-model
pilot did not beat the deployed specialists, and an eight-epoch plate/wrapper
follow-up failed retention checks. Neither was deployed. The pilot data, receipts,
previews and comparisons are external under `artifacts/orientation-training-20261028/`.

Condiment packet is retired from inference, evaluation and the active training
entry point. Its code, tests, checkpoints and previous configuration are preserved
externally under `history/packet-retired-20261027/`; earlier experiments and the
rejected long-run candidate remain under `artifacts/packet-focus/`.

```sh
python -B -m unittest discover -s tests -v
```

Only maintained source, configuration, focused tests and concise results are in
Git. Historical source/data/results and Git recovery copies remain outside the
repo. Git commit history is preserved.


## Disposal categories

A rules-based classifier converts each object detection into a disposal decision.
It adds no new neural model and does not infer packaging material or container
contents. Each detection in a mixed photo receives its own result.

```sh
./scripts/run_specialists.sh sort /path/to/image.jpg --device mps --output /tmp/disposal.json
```

The **user-selected experimental shortcut** in `configs/disposal.json` is:

| Detector class | Result at confidence >= 0.50 |
|---|---|
| Food | Compost |
| Disposable plastic drink bottle | Recycling, **without checking emptiness** |
| Milk/juice drink carton | Recycling, **without checking emptiness** |
| Plate, utensil, wrapper, hinged container | Review |

Detected objects below confidence 0.50 also require review; detections below
0.25 are omitted from the display. No object is automatically sent to landfill.
A bottle/carton prediction is an **object-recognition-based recycling suggestion**:
the system does not identify contents, material contamination or actual acceptance
of filled containers. Prior user confirmation covered **empty** milk/juice cartons
only. The `verified` configuration flag activates the user's requested rules; it
does **not** certify that every suggested item is locally recyclable.
[EPA guidance emphasizes local rules and material](https://www.epa.gov/recycle/how-do-i-recycle-common-recyclables).

Optional `--contexts` remains available for extra per-detection facts, but
`empty` is **not required** for bottle/carton recycling decisions in this policy.
The output records hashes, model configuration and per-object reasons.
Detection confidence is not disposal-bin confidence.

### Verified fixed development-cohort measurements (November 1, 2026)

The unchanged model predictions on **248 repaired development test photos**
contain **881 detections** at the 0.25 display threshold. A verified replay and fresh inference with the active four checkpoints both
produce these decisions at the unchanged 0.50 automatic threshold:

| Decision | Detection count |
|---|---:|
| Compost | 337 |
| Recycling (bottle + carton) | 173 |
| Review | 371 |
| **Total** | **881** |

Fresh MPS inference reproduced every frozen prediction exactly. All four checkpoint
bytes, active configuration metadata and all 248 image hashes were verified.
Of emitted detections, **57.89% receive an automatic suggestion and 42.11%
require review**. These percentages use emitted predictions, not labeled objects.
Detailed validation/test results and provenance are in
[verified measurements](results/sorting-verification.json).

On the same exposed development cohort, the prior **food-to-compost object
recognition proxy** measured **97.03% precision / 75.52% recall**. Separately,
**bottle-or-carton recognition at 0.50 confidence** measured
**93.64% precision / 84.82% recall** (162 correctly matched of 173
high-confidence predictions; 191 labeled bottle/carton objects).
These figures evaluate **object recognition**, not correct recycling of material
or independent three-bin classification accuracy. True three-bin accuracy
remains **unknown** without independently assigned bin labels.

The latest accepted detector change was a one-epoch bottle update. Its paired
validation AP50/AP50–95 rose from 77.77%/59.55% to 82.47%/62.00%; test
rose from 70.13%/56.86% to 75.27%/59.62%. The other specialists remain
unchanged. Detector scores are not affected by this disposal-rule update.

### Reproduce the sorting decision counts (no new YOLO inference)

The matching receipt is external
`artifacts/bottle-retention-20261031/test-active-confirmation.json`.
The older `artifacts/frozen-test-20261031/evaluation.json` predates the active
checkpoints and correctly fails configuration validation. Reproduce with a new
output path (existing outputs are never overwritten):

```sh
./scripts/run_specialists.sh replay \
  ../yolo-cafeteria-research-assets/artifacts/bottle-retention-20261031/test-active-confirmation.json \
  --expected-images 248 --verify-images \
  --manifest ../yolo-cafeteria-research-assets/artifacts/test-label-audit-20261031/audited-all248.json \
  --output /tmp/sorting-replayed.json
```

`--manifest` supplies file locations when receipt image names are not paths.
Replay validates receipt/config metadata; image bytes are verified only with
`--verify-images`. Checkpoint bytes were independently verified in this audit;
fresh inference also verifies them. The confirmed replay receipt is external
`artifacts/disposal-20261031/test248-no-empty-verified-20261101.json`.
Fresh inference and sorting receipts, joint matching details and the reproducible
measurement script are in that same folder with `20261101` filenames.

Class-specific matching confirms **489 correct object matches among 510 automatic
suggestions: 95.88% precision, 78.37% recall and 86.24% F1** for food/bottle/carton.
A separate **joint geometry-first** match of all emitted detections finds **483**
correct automatic object matches: **94.71% precision, 77.40% recall and 85.19% F1**.
The six differences arise when food/bottle boxes match overlapping wrapper labels
before checking object class. Neither method measures actual disposal eligibility.
Joint matching gives **52.92% automatic coverage of the 924 labeled objects**,
30.95% matched review and 16.13% missed; 21 automatic and 85 review predictions
are unmatched. See [methods, per-class metrics and limits](docs/sorting-evaluation.md).

### Weak-class optimization without regression

The active model is not altered automatically. For a candidate checkpoint,
evaluate it on exactly the same frozen 186-photo validation manifest, then run:

```sh
./scripts/run_specialists.sh guard results/baseline.json \
  /absolute/path/to/candidate-validation.json --target bottle \
  --output /absolute/path/to/bottle-retention.json
```

The candidate must improve the target class AP50–95 by at least 0.01
(one percentage point) and avoid any per-class or macro AP50/AP50–95 decline
greater than 0.002 (0.2 percentage points). A changed validation manifest
or image/label counts fails the gate. An optional historical 124-photo
retention check uses `--historical-baseline` and
`--historical-candidate`, supplied together. A guard pass does not deploy
weights or prove independent generalization. Do not tune on the 248-photo
test set; evaluate fresh holdout school captures after choosing a candidate.

Current priority is bottle recognition (33/43 matches at confidence >=0.50,
24/57 labeled bottles missed or below threshold). Food generalization to
fresh school photos remains unverified. Secondary diagnostic targets are
wrapper localization and utensil validation accuracy. Avoid raising AP
numbers by excluding difficult but valid samples or reusing test labels.

Use [the independent sorting evaluation protocol](docs/sorting-evaluation.md)
to collect human-verified bin labels. Existing test photographs may have been
exposed to training or model selection, and the repaired labels are AI-assisted,
so the displayed numbers are not independent generalization estimates.

The November 1 bottle trial was **rejected**: one classification-head-only epoch
on 527 training images changed validation bottle AP50/AP50–95 from
82.47%/62.00% to 77.95%/59.18%. Historical retention passed, but the primary
regression gate failed. No candidate test scoring or automatic replacement occurred.
See the [controlled comparison](docs/sorting-evaluation.md#controlled-bottle-candidate--rejected).
