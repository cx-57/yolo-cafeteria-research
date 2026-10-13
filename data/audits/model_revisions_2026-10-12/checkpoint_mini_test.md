# Uploaded 100-epoch YOLO checkpoint: mini-test findings (2026-10-12)

## Evidence and reproduction boundary

Inspected the user-supplied `yolo11n-corrected-baseline-100.zip`, containing `best.pt`, `last.pt`,
`results.csv`, validation contact sheets, confusion matrix, precision-recall curves,
and run `args.yaml`. ZIP SHA256:
`e6dba00eec237d364d39d6dbeb98c5a75a38e0ce88cf29197d946fde8a69366e`.
`best.pt` SHA256:
`33a1974dd37df1b00d847c95ea2982c429e3ae62903003a740bba7580d9ac5c1`.
The archive does **not** contain the 619 original validation images or their labels.
The saved model is the *old* 100-epoch checkpoint, predating Codex's additive LVIS corrections.
No change to GitHub source code or annotation files can improve these **unchanged weights** without a new optimization step.

## Actual native validation metrics (source: saved original run)

619 validation images; 1,472 labeled objects. Macro **mAP50 = 0.779**, **mAP50–95 = 0.573**.
Per-class metrics from the user's Ultralytics 8.4.175 console:

| Class | Validation images | Objects | P | R | AP50 | AP50–95 |
|---|---:|---:|---:|---:|---:|---:|
| food | 155 | 664 | 0.985 | 0.977 | 0.991 | 0.690 |
| plate | 103 | 216 | 0.689 | 0.483 | 0.559 | 0.392 |
| utensil | 38 | 79 | 0.787 | 0.841 | 0.851 | 0.602 |
| hinged_container | 47 | 47 | 0.989 | 1.000 | 0.995 | 0.787 |
| drink_carton | 351 | 371 | 0.961 | 0.960 | 0.977 | 0.795 |
| wrapper | 15 | 19 | 0.873 | 0.363 | 0.435 | 0.333 |
| bottle | 59 | 76 | 0.751 | 0.539 | 0.642 | 0.413 |

**Paper boats, napkins, condiment packets and bowls have no validation objects; their AP is undefined.**
The plot values may differ slightly from the second validation console pass due to rounding/validator configuration.

### Saved normalized confusion matrix

Approximate values read from the plotted matrix (**not** AP and using the confusion
matrix's own threshold/matching procedure):

- True plates: ~0.74 predicted plate, ~0.19 background.
- True wrappers: ~0.42 predicted wrapper, ~0.21 predicted bottle, ~0.32 background.
- True bottles: ~0.71 predicted bottle, ~0.09 predicted wrapper, ~0.16 background.

Wrapper/bottle confusion is particularly pronounced. The error mix is not adequately
explained by a generic "small YOLO model" hypothesis.

### Learning curve

`results.csv` contains all 100 completed epochs. Highest mAP50 = **0.78143** at
epoch 93; highest mAP50–95 = **0.57322** at epoch 100.
Last 10 epochs average mAP50 = **0.7789**. In epochs 61–80, the peak mAP50 was
**0.7775**, versus **0.7814** in epochs 81–100. This is weak marginal improvement
despite continued optimization; a third 100-epoch run on identical data has no
empirical justification.

## Inference mini-test: what was and wasn't tested

For a **smoke test only**, the checkpoint was safely loaded on CPU without
executing arbitrary pickle globals and forward inference was reconstructed using
the known YOLO11n layers. Predictions were obtained for 23 crops extracted
from the checkpoint's *saved, annotated* validation-preview mosaics.
Several recognizable plates were detected at confidence >0.8, and other difficult
scenes had no high-confidence plate detection. Annotation cyan overlays were
inpainted from the previews; some original content remains altered. Therefore
these tests are **not official Ultralytics inference on untouched photos,
cannot estimate per-class AP and cannot establish that corrected training helps**.
Keep any locally saved preview montages clearly labeled as reconstructed.

No independent NHS test, no corrected-data retraining, and no reliable new
plate/wrapper/bottle/napkin AP were performed. Real evaluation requires
the **original validation photos + generated corrected labels** alongside the
checkpoint in one runtime, or a separately trained candidate checkpoint for
a frozen-split A/B comparison.

## What to do next (without another 100-epoch run)

1. Review the existing baseline's predictions on **untouched** plate, bottle,
   wrapper and napkin photos. Keep heavily covered plates; review their boxes,
   scale, and degree of occlusion instead of deleting hard examples.
2. Validate and review newly appended LVIS food annotations and partial-label
   coverage semantics before training. Note these corrections expand validation
   food targets from 664 to 1,215 and cannot be compared directly with old AP.
3. Only if the model needs optimization, use **short fixed-split fine-tuning
   experiments from the existing best.pt** (e.g., 3–5 epochs) as a *screening
   experiment*, paired with an equal-budget control starting from the identical
   checkpoint. Short runs do not prove a 95% eventual mAP.
4. Test all 11 classes only after adding fully annotated **independent**
   validation scenes for napkins, bowls, condiment packets and paper boats;
   paper boats currently have zero training examples.
5. Any claim of `95% mAP50` must be made on a fully labeled, capture-group
   disjoint dataset with every claimed class represented. This benchmark does
   not currently exist.

**Conclusion:** the archived trained checkpoint still exhibits weak plate,
wrapper and bottle performance; this upload alone does **not** confirm
that the updated code has repaired those classes. Avoid claiming otherwise.
