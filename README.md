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

The primary repaired validation set scores **85.69% mAP50 / 64.90% mAP50–95**
with the active specialists. See [per-class results](results/baseline.json).
Labels are AI-reviewed with documented uncertainties, not human certified.

The release models score **87.05% mAP50 / 66.84% mAP50–95** on all
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
The current release scores **90.46% / 66.91%** on that historical 124-photo
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
`../yolo-cafeteria-research-assets/data/combined-seven-cafeteria-crops-20261031/data.yaml`.
It contains **2,332 training views, 186 validation photos and 248 test photos**.
Training contains 2,048 real photos plus 284 school variations. The school portion
is 300 views from **16 real training photos**, including nine reviewed, sharp
cropped-food photos admitted for training only. Soft targets and bin scenes stay
excluded. Exact quarter turns move boxes with the image; mild lighting variations
add no objects or camera viewpoints. Augmentation never crosses splits.
The dataset explicitly sets `allow_cropped_food_train: true`; the validator still
rejects cropped food in validation/test and boxes outside the image in every split.
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
