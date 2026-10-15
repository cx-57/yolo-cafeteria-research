# Expanded plate / wrapper / plastic-bottle experiment

Goal: at least 90% AP50 individually for every class; 85% individually is the user-approved fallback if progress is too slow. Neither threshold has been achieved. This supersedes earlier 613/619 recipes.

## What is included

All 3,096 original V2 image/label pairs were inventoried using current hash-validated approved overrides. The older directory named codex-reviewed had stale target rows for 176 of the 619 prior examples and was not used as the source of truth. The corrected candidate inventory initially contained 947 target-positive label files; this count includes disputed/false labels, not 947 verified positives.

The final dataset contains 900 distinct images: 813 verified positives plus 87 verified negatives. It includes all currently signed-off examples, including video frames, once each. It is not the prior 386-photo fine-tuning subset. After correction/adjudication, 129 target-positive candidates remain explicitly held; 2,067 other images lack complete target review and are not treated as negatives. Thus all candidates were considered, but disputed labels were not silently admitted to training.

| Split | Images | Verified negatives | Plate objects | Wrapper objects | Plastic beverage bottle objects |
|---|---:|---:|---:|---:|---:|
| train | 774 | 84 | 368 | 191 | 330 |
| val | 126 | 3 | 137 | 17 | 45 |

Plate policy: always one bounding box for an entire physical plate stack; distinct side-by-side plates remain separate. Bottle means plastic beverage bottles only. Corrections addressed glass/condiment bottles and a stick labeled as bottle, missing food packages, missing plates, padded bottle bounds, duplicate plate boxes, and plate-stack consistency. Existing verified non-plate boxes were preserved during stack review.

## Generalization and integrity

- All 900 image-byte and decoded-pixel hashes are distinct. No original-photo, exact-byte, decoded-pixel, meal-pair, or source-video group crosses train/validation.
- Paired school-lunch photos share a meal group. Frames from each waste video share a capture group. Actual V2 grouping is retained where possible; filename split prefixes do not assign splits.
- The original 92 comparison images, image hashes, decoded pixels, and all labels remain identical. Expanded validation has 126 images; its aggregate scores must not be compared directly with previous 92-image scores.
- The new recipe starts from original COCO-pretrained yolo11n.pt, not any focus fine-tuning checkpoint. One training view per image and standard installed Ultralytics augmentation avoid deliberate repetition of a few hard cases.
- This is still development validation, not an independent natural-cafeteria test. Repeated tuning can overfit validation; even 90% here would not prove 90% in deployment. Only 17 validation wrapper objects remain a limited estimate.
- The original 11-class dataset, training pipeline and original checkpoints were not edited.

## Latest trained experiment, before this expansion

The user-completed sampling640 run selected epoch 7. On the unchanged 92-image comparison set:

| Class | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
| plate | 81.19% | 68.54% | 76.99% | 59.34% |
| wrapper | 74.79% | 74.26% | 68.63% | 54.29% |
| bottle | 71.40% | 62.44% | 65.27% | 49.12% |
| Overall | 75.79% | 68.42% | 70.30% | 54.25% |

A one-epoch engineering smoke run on the intermediate 892-image dataset completed successfully in about 81 seconds of training plus setup/evaluation. It checks the pipeline, not accuracy convergence, and is not a candidate for deployment. No full training on the final 900-image dataset has run yet.

## Nine-minute screening run

This respects the short-run preference. Native Ultralytics time budgeting may finish the current epoch after the budget; setup/evaluation add time. A short cold-start run is not equivalent to completing 80 epochs and should not be expected to reach the target immediately.

```sh
cd /Users/Charles1/Desktop/yolo-cafeteria-research
.venv/bin/python scripts/focus_three_class.py train \
  --data data/focus-three-class-improvement/all-reviewed-stack \
  --model model/focus-three-class-improvement/user-expanded90-screen9 \
  --recipe model/focus-three-class-improvement/expanded90-screen9-recipe.json

.venv/bin/python scripts/focus_three_class.py evaluate \
  --data data/focus-three-class-improvement/all-reviewed-stack \
  --model model/focus-three-class-improvement/user-expanded90-screen9 \
  --eval-weights model/focus-three-class-improvement/user-expanded90-screen9/train/weights/best_min_ap50.pt
```

Evaluate the exact same checkpoint on the original comparison set:

```sh
.venv/bin/python scripts/focus_three_class.py evaluate \
  --data data/focus-three-class-613 \
  --model model/focus-three-class-improvement/user-expanded90-screen9 \
  --eval-weights model/focus-three-class-improvement/user-expanded90-screen9/train/weights/best_min_ap50.pt \
  --eval-name evaluation-comparison92
```

## Full 80-epoch convergence experiment

The full fresh-start run is the meaningful convergence experiment. At the measured smoke-epoch speed, expect roughly 1.5–2 hours on this Mac, with actual duration depending on workload and early stopping at 90% for every class. It has not been launched because longer jobs are for the user to execute. The 85% fallback is an acceptance threshold, not an automatic stopping condition in this recipe.

```sh
.venv/bin/python scripts/focus_three_class.py train \
  --data data/focus-three-class-improvement/all-reviewed-stack \
  --model model/focus-three-class-improvement/user-expanded90-full80 \
  --recipe model/focus-three-class-improvement/expanded90-recipe.json

.venv/bin/python scripts/focus_three_class.py evaluate \
  --data data/focus-three-class-improvement/all-reviewed-stack \
  --model model/focus-three-class-improvement/user-expanded90-full80 \
  --eval-weights model/focus-three-class-improvement/user-expanded90-full80/train/weights/best_min_ap50.pt

.venv/bin/python scripts/focus_three_class.py evaluate \
  --data data/focus-three-class-613 \
  --model model/focus-three-class-improvement/user-expanded90-full80 \
  --eval-weights model/focus-three-class-improvement/user-expanded90-full80/train/weights/best_min_ap50.pt \
  --eval-name evaluation-comparison92
```

Standalone evaluation writes full per-class precision, recall, AP50 and AP50–95 to evaluation/metrics.json. The native aggregate-selected best.pt can also be evaluated separately with a different --eval-name. Never combine per-class scores from different checkpoints to claim a threshold.

## Reproduce the final extraction

The destination must not already exist. The builder refuses existing output and original dataset directories.

```sh
.venv/bin/python scripts/focus_three_class.py build_all \
  --data data/focus-three-class-improvement/all-reviewed-stack \
  --one-box-per-stack \
  --review-files data/focus-three-class-improvement/expanded_review_combined_stack.json
```

Fourteen focused tests pass, including manual review hashes, duplicate/capture leakage, frozen image identity, plate-only explicit policy relabeling, and weakest-class checkpoint selection. The actual final build passed integrity checks; all original comparison labels remained unchanged despite the permitted documented policy update.

Final dataset fingerprint: 2f089e1570c5d6d0438760c55fe92dc6a641006c9f5d9fdd7a80e1dd3a080893
