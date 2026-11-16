# Shortened food-training trial results

The user requested a shorter run. The control completed two epochs; the stronger-scale candidate was interrupted during its first epoch and produced no completed checkpoint. No additional training was run, and nothing was deployed. The original weights and active configurations passed SHA256 preservation checks. Camera and servos stayed off.

## Fixed development comparison

Both models were evaluated on the same 186-image exposed development cohort at image size 640 on CPU. Predictions use confidence 0.001, NMS IoU 0.7 and max 300 boxes. The operating point stays at confidence 0.60 with class-aware one-to-one matching at box IoU 0.50. The control uses its last checkpoint. The parent prediction receipt completed before the pause and was verified and reused after resuming.

| Model | Food TP | Food FP | Food FN | Food precision | Food recall |
|---|---:|---:|---:|---:|---:|
| parent | 293 | 6 | 50 | 97.99% | 85.42% |
| control | 295 | 6 | 48 | 98.01% | 86.01% |

The control found two additional food targets, with the same six false food detections. Its food recall rose by 0.58 percentage points. This is a development result, with no demonstrated improvement on genuinely tiny food pieces.

| Class | Parent AP50 | Control AP50 | Parent AP50–95 | Control AP50–95 |
|---|---:|---:|---:|---:|
| food | 94.23% | 94.68% | 54.68% | 58.00% |
| plate | 81.32% | 80.16% | 55.47% | 55.90% |
| utensil | 40.75% | 37.78% | 20.86% | 21.47% |
| hinged_container | 99.50% | 99.50% | 94.50% | 94.50% |
| drink_carton | 94.12% | 93.24% | 55.70% | 57.83% |
| wrapper | 69.13% | 68.02% | 35.60% | 35.77% |
| bottle | 51.65% | 49.38% | 27.32% | 28.19% |

## Food by size

Medium food boxes: recall fell from 11/32 (34.38%) to 10/32 (31.25%). Large food boxes: recall rose from 282/311 (90.68%) to 285/311 (91.64%). There are zero genuinely small food boxes in this cohort. The overall food gain therefore provides no evidence of better small-piece detection.

## Decision

The control fails the predeclared preservation gate: AP50 declined for plate, utensil, drink_carton, wrapper and bottle. It remains an offline artifact; the original model stays selected.

The stronger-scale candidate has no completed checkpoint and was not evaluated. Its partial epoch was discarded. There is no completed matched comparison establishing a scale-augmentation benefit.

The cohort has provisional AI-assisted labels and prior development exposure. There are zero food boxes below 32² pixels at 640, so these scores cannot establish crumb or rice accuracy. No held-out-test images were scored, no confidence threshold was tuned, and no camera examples were used for training.

## Evidence

- [Comparison and preservation results](../runs/sorter/food-expert-offline/shortened-trial-comparison.json)
- [Parent prediction receipt](../runs/sorter/food-expert-offline/shortened-parent-evaluation.json)
- [Control prediction receipt](../runs/sorter/food-expert-offline/shortened-control-evaluation.json)
- [Size recall and wrong-class diagnostics](../runs/sorter/food-expert-offline/shortened-trial-diagnostics.json)
- [Run state](../runs/sorter/food-expert-offline/training-run-state.json)
- [Original plan and updated stop status](small-food-training-proposal.md)
