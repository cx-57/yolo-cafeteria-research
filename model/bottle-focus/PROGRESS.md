# Bottle specialist experiments — 2026-10-22

Target: at least **85% bottle AP50**. Best freshly verified result at this
snapshot: **82.6545%**, using a saved single-class YOLO11s checkpoint at
640 pixels with test-time augmentation and NMS IoU 0.5. The target has not been reached.

The fixed development validation set has **246 images and 67 bottle boxes**.
All validation images, bottle annotations and splits are preserved.
Source fingerprint:
`6668d2b7f0b16243640c14efdbb6bdffbb7f83fc880459befd58e8afcbfa2296`.
Validation protocol:
`29423591896198d41f5bd16392b51f06c7c3d7254f8a903b3dda74569cfe7de7`.
These are development results, with no independent test claim.

| Saved model / experiment | Ordinary AP50 | Augmented AP50 |
| --- | ---: | ---: |
| Pretrained YOLO11x, bottle output only | 40.21% | 45.65% |
| YOLO11s head fine-tune, best of eight epochs | 67.11% | 69.08% |
| YOLO11x head fine-tune, two completed epochs | 60.28% | 60.23% |
| YOLO11s context resampling, best of eight epochs | 68.39% | 71.33% |
| YOLO11s context resampling, best of 23 epochs | 73.67% | 71.22% |
| YOLO11s gentle feature adaptation, best of eight epochs | 82.08% | 82.14% |
| Same checkpoint, NMS IoU 0.5 | — | **82.65%** |

All scores are independently reloaded evaluations. The last detector’s
training-time validation score was 82.11%; the saved-model score above takes precedence.
Training validation uses a different batch size and saved checkpoints are
serialized at half precision, so the two scores need not match exactly.

The verified incumbent is retained locally at
`verified-incumbent/bottle.pt`; its settings and SHA-256 are recorded in
[`verified-incumbent/verification.json`](verified-incumbent/verification.json).
SHA-256: `edab60c58fb8a1e941557b4bd84955a5208b75b476e0a587854d3e679e7d1c63`.
Existing Git ignores exclude model weight files; reports and experiment code
are versioned, while checkpoints remain local.

Completed comparisons:

- Unfreezing later features at learning rate 0.0001 did not beat the
  initialization in three epochs.
- Inference at 960 and 1280 pixels did not beat 640-pixel augmented inference.
- A training-only BatchNorm recalibration pass changed no learned parameters,
  but reduced validation AP50 for both tested checkpoints.
- Training-only resampling repeats 124 existing bottle-positive context
  photos four times per epoch. Every other training photo remains included
  once. There are 1,056 unique training images, 1,428 exposures and 890 bottle
  exposures per epoch; no new photos were admitted.

Context resampling completed 23 epochs. Its epoch-16 averaged candidate
reached 72.73% augmented AP50. Gentle feature adaptation then started from
the verified 73.67% ordinary-inference checkpoint, retaining fixed BatchNorm
statistics and reducing the learning rate to 0.00001. It completed 13 epochs; epochs 9–13 did not improve on epoch 7.
At 832 pixels, ordinary/augmented AP50 were 74.72%/80.16%; 640 pixels
with tighter NMS reached 82.65%.

The new `external-data-v7` experiment adds 19 freshly downloaded LVIS
photos with 36 plastic beverage bottle boxes to training only. The
AI-assisted visual reviews are explicitly not human verification.
Validation remains fixed at 246 photos / 67 boxes. A separate 548-photo
Mendeley source was sampled and rejected as supplied after 24 photos
showed repeated scenes and incomplete/out-of-bounds annotations.
New-data training results are pending.

Run using the existing local runtime:

```sh
PYTHONPATH=/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/py312:$PWD/scripts BOTTLE_PHASE_EPOCHS=8 .venv/bin/python -u model/bottle-focus/gentle-feature-adaptation-v6/run.py
```

The cached dataset is prepared by
`validation-now/prepare_data_snapshot.py`. Individual experiment directories
contain their scripts, configurations, metrics, hashes and evaluation output.
