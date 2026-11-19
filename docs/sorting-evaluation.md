# Measuring three-bin sorting performance

**Actual three-bin accuracy has not yet been measured.** The repaired 248-photo
set has object labels, but not independent per-object disposal-bin labels.
The existing 35.39% metric only measures correct automatic compost assignments
as a fraction of all object boxes. It is not sorting accuracy.

## How to measure it

1. Collect a new, held-out cafeteria photo session. Do not reuse images that
   influenced training, validation, test evaluations or model selection.
2. Independently annotate each eligible item with a tight normalized XYXY box
   and the correct school disposal bin: compost, recycling or landfill.
   Determine material, contamination, reuse and whether containers are empty
   from evidence and school acceptance rules. Resolve uncertain cases manually.
3. Run the frozen sorting pipeline without changing thresholds or rules.
4. The current user-selected rule assigns detected bottles and drink cartons
   to recycling at confidence >= 0.50 **without checking emptiness**. This
   experimental shortcut is not proof that filled containers are recyclable
   locally. Preserve independent evidence of correct bin eligibility.
5. Preserve photos with zero labeled items, and use identical image path keys.

Run from the repository root:

    ./scripts/run_specialists.sh sort /absolute/path/to/new_photos \
      --device mps --output /absolute/path/to/sort_predictions.json

Create an independently checked labels file like this:

    {
      "images": {
        "/absolute/path/to/new_photos/image01.jpg": [
          {"box_xyxy_normalized": [0.10, 0.20, 0.40, 0.50],
           "category": "compost"},
          {"box_xyxy_normalized": [0.50, 0.20, 0.80, 0.70],
           "category": "recycling"}
        ],
        "/absolute/path/to/new_photos/empty_image.jpg": []
      }
    }

These labels/boxes are ONLY a format example. They are not actual photo
annotations. Ground truth cannot be generated from model predictions.
The image names must match those in the sorting JSON's decisions map exactly.

Then run:

    python3 scripts/evaluate_sorting.py \
      --predictions /absolute/path/to/sort_predictions.json \
      --truth /absolute/path/to/human_bin_labels.json \
      --output /absolute/path/to/sorting_scores.json

The evaluator matches predictions to labeled objects at box IoU >= 0.50,
highest confidence first. Matching does not look at predicted or true bins.

## Interpret the output

- correct_automatic_over_all_labeled_objects: correctly automatically sorted
  / all labeled objects; a coverage-sensitive overall success fraction.
- accuracy_among_matched_automatic_decisions: correct / (correct + wrong bin)
  among matched automatic decisions; excludes reviews, misses and unmatched
  false-positive predictions.
- precision_including_unmatched_auto_predictions: correct / all automatic
  bin assignments, including unmatched false positives and duplicates.
- automatic_coverage_of_objects: fraction of ground-truth objects receiving
  an automatic, matched compost/recycling/landfill decision.
- matched_review, missed_objects, wrong_bin_automatic, and
  unmatched_predictions must be reported alongside accuracy.
  Review is abstention, not a correct prediction.
- confusion_by_true_bin: actual bin vs predicted bin/review/missed.

For a research-paper claim, use independently collected photos with
human-verified disposal labels and document the local sorting policy.
Previously inspected or trained-on images are not an independent benchmark.
Do not claim complete three-bin accuracy until that evaluation exists.

To check implementation:

    python3 -m unittest discover -s tests -p 'test_sorting_evaluation.py' -v

## Verified development results — November 1, 2026

The local repository and GitHub `origin/main` both started at `4663c90b`, with a
clean working tree. Fetch succeeded; no merge, reset or stash was needed.
All four checkpoint SHA-256 values match `configs/specialists.json`, whose hash is
`c9b70ce27b55a27bd6ab70c4f837fbbe0e4269c58f97622ac586c4e89c26aa78`.
The correct 248-image receipt is external
`artifacts/bottle-retention-20261031/test-active-confirmation.json`, SHA-256
`01b8c6448b851b57f53768eff67fddefc6b4c927fb42603b4c1421c94c3e2848`.
The older `frozen-test-20261031/evaluation.json` is incompatible with the active
checkpoints; it was rejected, not relabeled as current.

Replay and fresh MPS inference both give **337 compost, 173 recycling,
371 review, zero landfill** (881 detections, display floor 0.25). Fresh predictions
are exactly equal to the frozen receipt, not merely equal in aggregate counts.
The repaired 248-image manifest and every original image hash were checked.
Food goes to compost; bottles/cartons recycle regardless of emptiness; plates,
utensils, wrappers and hinged containers require review. Scores below 0.50 require
review if displayed; scores below 0.25 are omitted. No policy thresholds changed.

The first `--verify-images` attempt failed because receipt names were basenames,
not filesystem paths. `replay --manifest ... --verify-images` now verifies exact
receipt name/hash membership, then hashes the manifest's original image locations.
Regression tests cover changed bytes and wrong cohorts.

All precision/recall/F1 values below are **object recognition** at confidence
>=0.50 and box IoU >=0.50, using confidence-descending one-to-one matches requiring
an exact object class. AP uses the existing native detector evaluator and its
matching convention across thresholds; AP and operating-point metrics are distinct.
No bin ground truth was fabricated from these object annotations.

### 248-photo test development cohort

| Class | Objects | TP / FP / FN | Precision | Recall | F1 | AP50 | AP50–95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| food | 433 | 327 / 10 / 106 | 97.03% | 75.52% | 84.94% | 83.91% | 55.09% |
| plate | 123 | 103 / 15 / 20 | 87.29% | 83.74% | 85.48% | 87.04% | 71.60% |
| utensil | 98 | 70 / 7 / 28 | 90.91% | 71.43% | 80.00% | 86.06% | 61.26% |
| hinged_container | 4 | 4 / 1 / 0 | 80.00% | 100.00% | 88.89% | 99.50% | 91.17% |
| drink_carton | 134 | 129 / 1 / 5 | 99.23% | 96.27% | 97.73% | 97.55% | 74.28% |
| wrapper | 75 | 58 / 7 / 17 | 89.23% | 77.33% | 82.86% | 85.13% | 57.64% |
| bottle | 57 | 33 / 10 / 24 | 76.74% | 57.89% | 66.00% | 75.27% | 59.62% |

All seven classes, micro precision/recall/F1: **93.42% / 78.35% / 85.23%**. Macro mAP50/mAP50–95: **87.78% / 67.24%**.

### 186-photo frozen validation cohort

| Class | Objects | TP / FP / FN | Precision | Recall | F1 | AP50 | AP50–95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| food | 343 | 306 / 6 / 37 | 98.08% | 89.21% | 93.44% | 95.77% | 65.01% |
| plate | 110 | 85 / 12 / 25 | 87.63% | 77.27% | 82.13% | 81.56% | 68.99% |
| utensil | 101 | 67 / 10 / 34 | 87.01% | 66.34% | 75.28% | 74.94% | 46.77% |
| hinged_container | 2 | 2 / 0 / 0 | 100.00% | 100.00% | 100.00% | 99.50% | 94.50% |
| drink_carton | 94 | 88 / 6 / 6 | 93.62% | 93.62% | 93.62% | 92.84% | 65.19% |
| wrapper | 85 | 56 / 6 / 29 | 90.32% | 65.88% | 76.19% | 77.47% | 54.31% |
| bottle | 42 | 34 / 8 / 8 | 80.95% | 80.95% | 80.95% | 82.47% | 62.00% |

All seven classes, micro precision/recall/F1: **93.00% / 82.11% / 87.22%**. Macro mAP50/mAP50–95: **86.36% / 65.25%**.

### Combined automatic suggestions and denominators

Class-specific food/bottle/carton matching confirms 489 TP, 21 FP and 135 FN:
**95.88% precision / 78.37% recall / 86.24% F1**. The denominator for recall is
624 food/bottle/carton labels. Other object classes remain review-only.

In a separate joint geometry-first pass, all 881 displayed detections compete for
924 labeled objects, highest confidence first, without consulting class or category
when assigning a box. This finds **483 exact-class automatic matches, six automatic
wrong-object matches, 21 unmatched automatic predictions, 286 matched review,
85 unmatched review and 149 missed objects**. Five food boxes and one bottle box
match overlapping wrapper labels. This explains why the class-specific sum is
optimistic relative to this joint matching convention; these are matching-policy
differences, not six newly established wrong-bin labels.

The joint object-recognition proxy is **94.71% precision / 77.40% recall /
85.19% F1** (483/510 precision, 483/624 recall). It still does not establish that
recognized items qualify for their proposed bin. Automatic geometry coverage is
489/924 = **52.92% of labeled objects**; correct exact-class automatic matches are
483/924 = **52.27%**. Matched review is 286/924 = **30.95%**, and missed objects
are 149/924 = **16.13%**. Among emitted predictions, automatic coverage is
510/881 = **57.89%** and review rate is 371/881 = **42.11%**. Predictions below
0.25 are omitted, so these last rates do not count every real item.

### Provenance and remaining limits

The frozen validation manifest embeds corrections that are absent from 109 of its
186 referenced label files, including ten with bottle-label differences. The files'
recorded hashes still match. Evaluation has always used the embedded manifest rows;
this audit preserves that exact manifest hash and does not substitute disk labels.
Candidate validation copies those frozen embedded rows. Training labels were checked
against their own manifest and copied separately. Original annotations remain intact.

The annotations are AI-assisted, not human certified. Both development cohorts
have inherited checkpoint exposure and repeated inspection. Hinged-container support
is only two validation and four test objects. Independent school generalization,
material/contamination/emptiness recognition, actual local bin eligibility, landfill
performance and actual three-bin precision/recall/F1 remain unverified. Independent
human bin labels and new capture days are required to resolve these gaps.

Compact results: [sorting-verification.json](../results/sorting-verification.json).
Detailed receipts and `verify-20261101.py` are external under
`artifacts/disposal-20261031/`. `evaluate` now includes reproducible fixed-threshold
per-class counts and micro metrics in its `operating_point` field.

## Controlled bottle candidate — rejected

One fixed epoch trained only the bottle classification head (`model.23.cv3.*`),
seed 42, AdamW learning rate 0.00002, batch 1, nominal batch 8, with mild color
augmentation and no geometric augmentation. All backbone/box-head state and all
BatchNorm statistics were verified unchanged at checkpoint precision. The run used
271 bottle-positive and 256 deterministically sampled negative training images.
Forty-eight training views sharing recorded identity/group keys with the frozen
validation/test or historical reference were excluded before selection. Validation
and historical labels/membership/scoring remained frozen. Historical inherited
exposure in the starting checkpoint is still possible.

| Cohort | Bottle AP50 before → after | Bottle AP50–95 before → after | Gate |
|---|---:|---:|---|
| Frozen validation, 186 photos | 82.47% → 77.95% | 62.00% → 59.18% | Failed |
| Historical reference, 124 photos | 97.10% → 98.00% | 82.82% → 83.71% | Retained |

At confidence 0.50, bottle validation precision/recall/F1 fell from
80.95%/80.95%/80.95% to 71.05%/64.29%/67.50%. Primary macro AP50/AP50–95 fell
from 86.36%/65.25% to 85.72%/64.85%; all other classes were unchanged.
The predeclared gate required at least +1 percentage point bottle AP50–95 and
no per-class or macro regression greater than 0.2 percentage points on either
benchmark. `guard` exited **2**, correctly rejecting this candidate. Historical
retention alone does not justify promotion. There is **no demonstrated new model
improvement** from this trial. The active four checkpoints are intact.

The candidate was never scored on test, and no follow-up hyperparameter search
was performed. Its protocol, input hashes, frozen-state checks, copied train/val
data, separate checkpoint, paired evaluations and guard report remain external in
`artifacts/bottle-candidate-20261101/`. The complete comparisons are also in
[sorting-verification.json](../results/sorting-verification.json).

Reverification repeated the 248-image native inference, replay, metrics and candidate
retention gate, and all 53 tests passed again. All 1,436 recorded protected hashes
match. There is no pre-task inventory of every historical asset, so this audit
does not independently certify the entire archive unchanged. The detailed repeat
report is external `artifacts/disposal-20261031/reverification-20261101.json`.
