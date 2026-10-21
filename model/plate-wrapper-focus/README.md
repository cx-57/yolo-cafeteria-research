# Plate and wrapper: corrected benchmark and YOLO11n training

The current requested minimum is **88% mean AP50 across plate and wrapper**, preferably 90%. Precision, recall and AP50–95 are reported separately. The original 11-class dataset, pipeline and weights are preserved. Bottle is outside the active scope.

## Current selected model

The lower-learning-rate continuation `lr-low` has recorded 28 completed epochs and selected epoch 17. A fresh inference pass on all 246 corrected validation images reproduced **90.22% mean AP50**. Both classes also individually exceeded 90% AP50. The requested 88% minimum and preferred 90% mean are met on development validation.

| Class | Precision | Recall | AP50 | AP50–95 | Objects |
|---|---:|---:|---:|---:|---:|
| Plate | 90.60% | 81.00% | 90.05% | 76.09% | 238 |
| Wrapper | 95.98% | 86.17% | 90.38% | 65.79% | 94 |
| Mean | 93.29% | 83.58% | 90.22% | 70.94% | 332 |

Fresh verification, checkpoint checksum, dataset fingerprint and validation protocol are recorded in `lr-low/verified_average90.json`. The preserved checkpoint is `lr-low/train/weights/verified-epoch17-f7055a6bddfa.pt`; its SHA256 is `f7055a6bddfaa2e84480afd7c875571333f7cb4e30b941236eb57857df85d8f1`. Future training can update `best_min_ap50.pt` without replacing this snapshot. The older epoch-12 snapshot and `verified_average88.json` remain historical results. AP50 is not object recall: missed plates remain a limitation.

After correcting four annotation errors, the original-rate run fluctuated and selected epoch 5. A matched one-epoch comparison from that exact checkpoint retained the optimizer/EMA and changed only learning rates: scaling rates by 0.3 produced 88.12% mean AP50 versus 86.53% for the unchanged-rate control. See `learning_rate_comparison.json`. The lower-rate run subsequently recorded 28 completed epochs, keeping the same images, labels, augmentation and 80-epoch horizon. This supports the lower rate for this experiment; it does not establish a universal optimum.

These are development-validation results, repeatedly used for tuning. No independent new-session cafeteria test is available.

## Current dataset

`data/plate-wrapper-focus/reviewed-v2` retains all **1,302 images** and the existing capture-grouped splits. No validation images were removed or moved into training.

| Split | Images | Plate objects | Wrapper objects | Verified negatives |
|---|---:|---:|---:|---:|
| Train | 1,056 | 650 | 395 | 368 |
| Validation | 246 | 238 | 94 | 48 |

A targeted visual audit of 50 inherited plate-source reviews identified four images requiring corrections: an occluded plate incorrectly treated as a validation negative; a paper food boat labeled plate with its flexible food package omitted; a missing serving platter; and incomplete plate/wrapper extents. Three training images and one validation image were corrected with image/label hash-bound reviews. Paper boats remain a separate excluded class. Identifiable clipped or occluded plates remain positives. Wrapper boxes enclose the whole package.

Evidence: `label-audit-20261018/corrections-before-after.jpg`, `label-audit-20261018/report.json`, and `data/plate-wrapper-focus/label-corrections-v2.json`.

The earlier `capture-grouped-clean` two-class dataset has a quality hold. Its saved scores are historical measurements on superseded labels. The earlier validity audit verified structural integrity and reproduced inference; it did not detect these semantic label errors.

## Re-evaluated starting models

Both preserved models were evaluated on the exact same corrected validation images/labels, before new training:

| Model | Plate AP50 | Wrapper AP50 | Overall AP50 |
|---|---:|---:|---:|
| Original three-output model, bottle filtered | 80.78% | 86.35% | 83.56% |
| Previous two-class epoch1 | 83.27% | 83.18% | 83.23% |

Full precision, recall, AP50 and AP50–95 are in `rebaseline-3a92a28252f4/comparison.json`. New training initializes from the previous two-class epoch1 checkpoint. Changes caused by revised validation labels are not training improvements.

## Historical corrected-data checkpoint after two epochs

Separate validation: plate precision 88.70%, recall 75.84%, AP50 84.46%, AP50–95 68.65%; wrapper precision 96.22%, recall 77.66%, AP50 86.95%, AP50–95 56.57%. Overall AP50 is 85.71%, AP50–95 is 62.61%. The 90% per-class AP50 target has not been reached.

The same checkpoint scored 86.83% plate / 83.27% wrapper AP50 on unaugmented training images. This does not show a large positive training/validation gap at this checkpoint; it also does not establish freedom from overfitting or real-cafeteria performance. Full diagnostics are in `two-class90-reviewed/training_fit_diagnostic.json`. A continuation retains the optimizer and full learning-rate horizon.

## Commands

From the repository root, build a new corrected dataset once, then verify and re-evaluate the preserved baselines:

```bash
.venv/bin/python scripts/plate_wrapper.py build --rebaseline
.venv/bin/python scripts/plate_wrapper.py verify
.venv/bin/python scripts/plate_wrapper.py baseline
```

The corrected dataset is already built; do not rerun `build` into its existing directory. Verification checks source provenance, exact membership, hashes, decoded pixels, box geometry, target-label preservation, reviewed negatives, original/capture/scene grouping and all declared validation corrections. Known quality holds block use. Unrecognized related captures can still escape the recorded grouping. Training on corrected labels requires a matching rebaseline report. A warm-start membership check ensures the prior two-class run had the same image/split assignments.

The development-validation target is met; further training is optional. To continue the selected lower-rate experiment toward the **80-epoch total schedule**, retaining its optimizer and scheduler:

```bash
.venv/bin/python scripts/plate_wrapper.py train --resume --model model/plate-wrapper-focus/lr-low --epochs 80
```

Or continue in approximately nine-minute training chunks:

```bash
.venv/bin/python scripts/plate_wrapper.py train --resume --model model/plate-wrapper-focus/lr-low --epochs 80 --minutes 9
```

Epochs 1–28 of this corrected-data schedule are already complete; this resumes at epoch 29, with up to 52 remaining. It does not start 80 additional epochs. Early stopping remains enabled: 15 epochs without improvement in native validation fitness, or both classes reaching 90% AP50 during training validation. Thus the run may finish before epoch 80. The callback's per-class 90% stopping rule is stricter than the current requested mean-88% reporting threshold.

The time budget is checked after complete epochs and may overrun by an epoch. Preflight checks, final validation and integrity checks add time. Resuming keeps the optimizer and the same 80-epoch learning-rate horizon. Initial corrected-data training used a new AdamW optimizer at learning rate 0.001, three warmup epochs, batch 4, 640 pixels, MPS, seed 42 and standard Ultralytics augmentation. The selected lower-rate branch retains that optimizer/EMA and scales its base rate to 0.0003. All training images are retained without bottle replay.

Training saves the checkpoint with the strongest weaker-class AP50 and separately evaluates it. Report/recheck it with:

```bash
.venv/bin/python scripts/plate_wrapper.py evaluate --model model/plate-wrapper-focus/lr-low
```

Current outputs are in `model/plate-wrapper-focus/lr-low/`; the earlier original-rate comparison remains in `two-class90-reviewed/`:

- `evaluation.json`: separately evaluated per-class and overall precision, recall, AP50 and AP50–95; deltas use the initial model on corrected labels.
- `minimum_class_selection.json`: selected epoch and checkpoint hash.
- `train/results.csv`: learning curve.
- `train/weights/best_min_ap50.pt`: selected model.
- `train/weights/last_resume.pt`: optimizer-bearing checkpoint for continuation.
- `verified_average90.json`: fresh verification of the preserved epoch-17 snapshot; the preferred mean-90% goal and both per-class AP50 values exceed 90%.
- `verified_average88.json`: historical fresh verification of the preserved epoch-12 snapshot against the mean-88% goal.

Evaluations are bound to checkpoint hashes. Matching saved evaluations can be reused; newer checkpoints receive separate folders. The original filtered inference configuration remains a preserved baseline and is not automatically replaced by a new model.

## What these scores establish

These are **development-validation** results. The same 246-image validation set has guided model selection repeatedly. A separate set of newly collected cafeteria photos, grouped by capture session and withheld from training and tuning, is still needed for an independent test. Pretraining overlap with public-source images and unrecognized related captures have not been independently excluded. The targeted label audit does not certify every annotation in all 1,302 images.

AP50 is not overall accuracy or recall. Report missed objects and false alarms as well as AP50. Neither reaching a validation target nor correcting labels establishes new-cafeteria performance.
