# Live detection comparison — November 15, 2026

The user reported utensils fluctuating between utensil/wrapper and bread not
being detected as food. The live application initially used an older unified
seven-class YOLO11n. No retraining was performed. Original datasets, labels,
checkpoints, frozen specialist config and historical result files were unchanged.

## Changes

- Live and annotated-image labels now require the configured confidence
  threshold (default .60). Previously the display showed every candidate down
  to the .15 occupancy threshold. This is a display/decision consistency fix,
  **not a change to model accuracy**. Low-confidence boxes still participate in
  ambiguity, occupancy and clear/rearm checks.
- `--detector specialists` explicitly loads the four existing frozen custom
  models and retains all seven output classes. The unified YOLO11n remains the
  default and is selectable with `--detector unified`.
- Native class IDs are validated before remapping with the existing research
  helper. Invalid classes/geometry, changed checkpoint hashes and invalid native
  resolutions abort instead of being silently changed or omitted.
- Models remain loaded. CPU inference uses two Torch threads to avoid excessive
  thread overhead in this process. The two-second freshness watchdog is retained.

Try the alternative in detection-only mode:

```sh
python3 scripts/live_sorter.py --detector specialists --mode detection --camera 0
```

This uses the repository's frozen specialists, including the larger utensil
model at its native 800-pixel resolution. It is not a new seven-class nano
checkpoint, and takes more computation than the single YOLO11n. Hardware modes
still require verified disposal rules, physical bin arrangement and ROI; no
movement is enabled by selecting a detector.

## Comparison method

All variants used the same **186-image / 777-object** manifest:
`../yolo-cafeteria-research-assets/artifacts/validation-label-audit-20261030/audited-eligible.json`.
SHA256: `0e8e8afef975539943f2ab39e0f402cce66ca25fbd0292ff3318246a6bbad1ec`.
Image hashes were verified before inference. Inference used CPU, confidence .001
for AP, class-aware NMS IoU .70, maximum 300 boxes and the installed Ultralytics
8.4.80. Fixed operating metrics use confidence **.60**, box IoU **.50**, and
confidence-ordered one-to-one matches requiring the exact ground-truth class.

**These are exposed development images with provisional AI-assisted labels,
not an independent test or live-camera accuracy estimate.** Models have inherited
training/development exposure. No original split or label was changed. No reserved
test cohort was used for this comparison. Live performance needs fresh examples.

| Existing detector | AP50 | Precision at .60 | Recall at .60 | Correct / false / missed |
| --- | ---: | ---: | ---: | --- |
| Configured unified, 640 | 75.81% | 95.49% | 65.38% | 508 / 24 / 269 |
| Same unified, 800 | 70.51% | 91.91% | 59.97% | 466 / 41 / 311 |
| Earlier combined pilot, 640 | 15.48% | 60.78% | 7.98% | 62 / 40 / 715 |
| Earlier overnight parent, 640 | 55.11% | 88.15% | 23.94% | 186 / 25 / 591 |
| Four frozen specialists | 86.36% | 94.04% | 79.15% | 615 / 39 / 162 |

For the reported problem classes, at the same .60 threshold:

| Class | Unified correct / objects | Specialists correct / objects | Unified false | Specialist false |
| --- | ---: | ---: | ---: | ---: |
| Food | 293 / 343 | 299 / 343 | 6 | 4 |
| Utensil | 8 / 101 | 58 / 101 | 1 | 9 |
| Wrapper | 42 / 85 | 56 / 85 | 3 | 4 |

The specialists improve detection coverage and development F1 (.7762 → .8595),
with a **1.45 percentage-point reduction in overall precision**. They do not
universally improve every error type. Utensil precision at this threshold is
86.57%, versus 88.89% for the unified model's very small set of nine utensil
predictions. This does not prove live utensil↔wrapper confusion is eliminated.
Food coverage improved modestly; this does not prove a particular bread item
will be detected.

A diagnostic wrong-class count reserves correct same-class matches first, then
matches remaining predictions to unmatched truth at IoU .50. This avoids calling
valid nested food/packaging detections class errors. At .60 the unified had one
such class error, and specialists zero; their remaining false detections were
unmatched boxes. The count depends on this matching convention and provisional
annotations. It is not a field misclassification rate.

Additional uniform specialist thresholds .65/.70/.75/.80 were checked from the
same cached predictions. Raising the threshold to .80 improved aggregate
precision to 95.96% but reduced utensil detections to 12/101 and food detections
to 269/343. This was **not selected**, because it sacrifices most of the utensil
benefit and worsens the missing-food concern. No optimal threshold is claimed.
The previous research class-specific threshold policy remains undeployed and
unchanged.

## Evidence and verification

Local, Git-ignored receipts are in `runs/sorter/model-comparison-20261115/`:

- Four unified-variant JSON files and `summary.json`, with checkpoint/manifest
  hashes, every prediction, per-class AP and operating metrics.
- `frozen-specialists.json`: complete original predictor receipt, exact model
  specs/hashes and scores. It reproduced the frozen 86.36% AP50 / 65.25% AP50–95
  development summary.
- `specialist-threshold-check.json`: explicitly labeled development-only
  threshold tradeoffs.
- `compare-unified.py` and `compare-specialists.py`: local comparison scripts.
  Their asset paths target this Mac; the specialist script invokes its sibling
  unified-comparison script and reuses cached results.
- `live-adapter-verification.json`: actual live-adapter parity checks on five
  labeled utensil images plus a local camera frame; inference durations included.
  These six checks took .71–.97 seconds per frame on CPU. This is a small
  timing sample, not a guaranteed camera FPS or worst-case bound.

The full repository suite after changes ran **192 tests: 191 passed, 1 existing
skip** (unavailable conversation bundle). New regressions verify specialist
class ownership/remapping, checkpoint hashes, malformed native class rejection,
native-resolution validation, display filtering and retention of hidden
occupancy detections. Python compilation and whitespace checks passed.

Camera-based bread/utensil tests remain necessary. If an actual object stays
missed or confidently mislabeled, capture representative platform views,
independently label them, and reserve separate evaluation examples before any
training. Training a separate candidate would require explicit authorization;
it must not overwrite the current checkpoints or alter original research data.

The first specialist live launch timed out during camera startup. A separate
camera0 probe then opened at1280×720, and the subsequent specialist live
launch stayed running without a reported inference/watchdog error during the
startup observation. This establishes camera startup for that run, not
verified bread/utensil classification. No Arduino port was opened.
