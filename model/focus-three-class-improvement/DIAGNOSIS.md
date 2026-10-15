# Three-class experiment: next run

Target: AP50 >= 0.80 individually for plate, wrapper, and plastic beverage bottles. This is an object-detection metric, not classification accuracy. Target not yet achieved.

## Current evaluated result

Checkpoint: stills-ft8/train/weights/best.pt, native validation at 640.

| Class | Precision | Recall | AP50 | AP50–95 | Objects |
|---|---:|---:|---:|---:|---:|
| Plate | 86.19% | 56.09% | 76.11% | 56.19% | 89 |
| Wrapper | 69.93% | 87.50% | 82.75% | 68.61% | 8 |
| Bottle | 74.83% | 55.56% | 58.78% | 44.71% | 36 |
| Overall | 76.98% | 66.38% | 72.55% | 56.50% | 133 |

Only eight validation wrappers make that class estimate fragile. This validation set has been used for tuning; it is not an independent test set.

## Underlying findings

- Bottle AP50 is 97.07% on the 386 unique still training photos versus 58.78% on validation: a generalization gap.
- 78% of validation bottles occupy less than 4% of their image, versus 39% of training bottles. Multiple-bottle photos are about 21% of bottle validation images versus 5% of training images.
- At confidence 0.25, bottle failures included five localization errors, four low-confidence detections, three undetected objects, and one wrong-class detection. Overlapping bottles sometimes merge into one prediction.
- Higher-resolution inference alone, tiled inference, and the interrupted higher-resolution training probe did not meet the target. No broad annotation error was established by the inspected examples; the suspected Sparkling ICE bottle is plastic according to its approved review.
- Native best.pt selection optimizes an aggregate score. The script now also saves best_min_ap50.pt by the weakest class, preventing a strong class from hiding a weak one.

## Dataset and proposed intervention

The original accepted 613-image experiment is retained. A separate 619-image pool adds six reviewed school-lunch training photos with approved corrections. Its 92 validation images and labels are byte/row identical to the original validation set. Integrity checks passed for boxes, verified negatives, canonical original grouping, exact image bytes, and decoded pixels.

Next recipe uses 386 reviewed still training photos, 760 augmented exposures: three exposures for wrappers and small/crowded/edge-clipped plastic bottles, one otherwise. Unique training objects: plate 204, wrapper 76, bottle 226. Validation objects: plate 89, wrapper 8, bottle 36. Repetition adds no independent examples. Video frames remain in the audited pool and were seen by inherited checkpoints, but are omitted from this fine-tuning stage.

The recipe tests 960-pixel input, stronger scale variation, modest mosaic augmentation, and a frozen backbone section. It starts from the existing stills checkpoint. This is a hypothesis, not a proven improvement. If it fails, the evidence favors additional independent, reviewed small/crowded plastic-bottle photos over simply adding epochs.

## Run locally (approximately nine-minute training budget, plus setup/evaluation)

```sh
cd /Users/Charles1/Desktop/yolo-cafeteria-research
.venv/bin/python scripts/focus_three_class.py train \
  --data data/focus-three-class-improvement/reviewed-619 \
  --model model/focus-three-class-improvement/user-hard-cases-80 \
  --recipe model/focus-three-class-improvement/next-hard-cases-recipe.json

.venv/bin/python scripts/focus_three_class.py evaluate \
  --data data/focus-three-class-improvement/reviewed-619 \
  --model model/focus-three-class-improvement/user-hard-cases-80 \
  --eval-weights model/focus-three-class-improvement/user-hard-cases-80/train/weights/best_min_ap50.pt \
  --eval-imgsz 960
```

Training validates integrity before starting, uses MPS, and refuses an existing output directory. It stops early if every class reaches AP50 0.80; otherwise the time budget ends the run (an epoch may overrun it). Confirm success using the separate evaluation/metrics.json, not just the training callback. Selection details are in minimum_class_selection.json. Do not resume into an original checkpoint directory.

Rebuild the original strict 613-image dataset only into a new directory:

```sh
.venv/bin/python scripts/focus_three_class.py build --data data/focus-three-class-613-rebuilt
```

All eight focused tests pass, including protected recipe options and weakest-class checkpoint selection. A one-epoch engineering smoke test exercised the callback in installed Ultralytics; it is not an accuracy experiment. Original 11-class dataset, checkpoints, and pipeline were not edited.

## User hard-case run follow-up (2026-10-14)

The nine-minute run completed six epochs. Its weakest-class-selected checkpoint was epoch 2; saved score and evaluation checkpoint SHA256 match (425f268f475d36b72d50019bfc9b9777baacad08c7d5deb3c059141f41b36768).

| Evaluation | Plate AP50 | Wrapper AP50 | Bottle AP50 | Overall AP50 |
|---|---:|---:|---:|---:|
| New checkpoint at 960 | 68.25% | 60.61% | 64.25% | 64.37% |
| Same checkpoint at 640 | 75.69% | 72.58% | 63.99% | 70.75% |

The source checkpoint at 960 had approximately 65.4%, 66.0%, 68.3% AP50; the new run therefore did not improve every class at matched resolution. It is not a replacement for the current overall-best checkpoint. The multi-change recipe cannot identify the independent effects of freezing, augmentation, sampling, or training resolution.

Additional source-domain evidence: 96 of 226 still-training bottle objects (42.5%) come from dry_wet_waste, whereas validation contains none from that source. Trash_detection supplies 113/226 training objects (50%) but 31/36 validation objects (86.1%). This reinforces the source/size/context mismatch; it does not justify removing difficult validation images. All metrics continue to use the original 92 validation images.

A matched two-epoch sampling comparison uses the same starting checkpoint, optimizer, learning-rate schedule, image size, augmentation, seed, and validation set. Only the training list differs (526 versus 760 exposures). More exposures also mean more optimizer updates; this tests the practical sampling policy, not an equal-update causal effect.

## Completed sampling-only comparison

All checkpoints below were independently evaluated at 640 on the unchanged validation set. Two training epochs took 111 seconds for control and 153 seconds for sampling (excluding setup/final evaluation).

### ablation-control2: weakest-class-selected checkpoint

| Class | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
| plate | 82.23% | 67.42% | 76.71% | 54.75% |
| wrapper | 56.27% | 75.00% | 75.52% | 61.23% |
| bottle | 82.46% | 52.78% | 58.47% | 42.68% |
| Overall | 73.65% | 65.06% | 70.23% | 52.89% |

Checkpoint SHA256: bed0613316bb11f6861e114f55d87ea786c5fe9c1866ab5261c50109cad2a741

### ablation-sampling2: weakest-class-selected checkpoint

| Class | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
| plate | 83.91% | 70.79% | 80.66% | 58.11% |
| wrapper | 73.74% | 75.00% | 74.12% | 63.72% |
| bottle | 71.05% | 47.74% | 61.58% | 45.98% |
| Overall | 76.23% | 64.51% | 72.12% | 55.94% |

Checkpoint SHA256: 185d26b31d568d4c90ca72fcc9f5aaf2d636a8d94a6185157ede208df8349bad

The sampling run selected epoch 2 for a tiny bottle AP50 advantage, despite lower wrapper AP50 than epoch 1. This shows checkpoint selection sensitivity; it is not evidence of a large bottle improvement. The target remains unmet. A nine-image review queue is at data/focus-three-class-improvement/bottle_training_review_queue.csv; these are actual V2 train images, outside the existing pool by canonical group, byte hash and decoded pixels, and remain excluded pending complete three-class review.

### Sampling run: native aggregate-selected best.pt

| Class | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
| plate | 76.16% | 68.22% | 79.66% | 58.09% |
| wrapper | 64.65% | 87.50% | 81.89% | 67.59% |
| bottle | 75.41% | 61.11% | 61.48% | 47.29% |
| Overall | 72.08% | 72.28% | 74.34% | 57.66% |

This improves aggregate AP50 from the earlier stills checkpoint (72.55% to 74.34%), but does not meet 80% in every class. Scores from different checkpoints must not be combined. Validation has been used repeatedly for tuning; a future independent test is needed to assess generalization beyond it.

## Next recommended user run (supersedes the earlier 960-pixel command)

Extend the simpler sampling policy from the same original stills checkpoint for an approximately nine-minute training budget. This is a longer test of the policy, not a guaranteed route to 80%.

```sh
cd /Users/Charles1/Desktop/yolo-cafeteria-research
.venv/bin/python scripts/focus_three_class.py train \
  --data data/focus-three-class-improvement/reviewed-619 \
  --model model/focus-three-class-improvement/user-sampling640-80 \
  --recipe model/focus-three-class-improvement/next-sampling640-recipe.json

.venv/bin/python scripts/focus_three_class.py evaluate \
  --data data/focus-three-class-improvement/reviewed-619 \
  --model model/focus-three-class-improvement/user-sampling640-80 \
  --eval-weights model/focus-three-class-improvement/user-sampling640-80/train/weights/best_min_ap50.pt

.venv/bin/python scripts/focus_three_class.py evaluate \
  --data data/focus-three-class-improvement/reviewed-619 \
  --model model/focus-three-class-improvement/user-sampling640-80 \
  --eval-name evaluation-macro-best
```

Recipe path/subset/class checks passed. Both ablations and all evaluations completed; no training remains running. Eight focused tests pass. The next recipe adds no new annotations and changes no original files.
