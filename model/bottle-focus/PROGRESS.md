# Bottle specialist experiments — 2026-10-22

Target: at least **85% bottle AP50**. Best freshly verified result at this
snapshot: **71.3261%**, using a saved single-class YOLO11s checkpoint at
640 pixels with test-time augmentation. The target has not been reached.

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
| YOLO11s context resampling, best of eight epochs | 68.39% | **71.33%** |

The last row is independently reloaded evaluation. Its training-time
validation score was 69.20%; the saved-model score above takes precedence.
Training validation uses a different batch size and saved checkpoints are
serialized at half precision, so the two scores need not match exactly.

The verified incumbent is retained locally at
`verified-incumbent/bottle.pt`; its settings and SHA-256 are recorded in
[`verified-incumbent/verification.json`](verified-incumbent/verification.json).
SHA-256: `c82beeff12003d53264e66eeb63e6126d59cb26856e839aeb526c53100e6d210`.
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

The resampling run has resumed after epoch 8 with its optimizer intact.
Later epoch parameters will also be averaged as an additional saved-model
candidate. Neither the averaging result nor later epochs are claimed here.

Run using the existing local runtime:

```sh
PYTHONPATH=/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/py312:$PWD/scripts BOTTLE_PHASE_EPOCHS=8 .venv/bin/python -u model/bottle-focus/context-resampling-v5/run.py
```

The cached dataset is prepared by
`validation-now/prepare_data_snapshot.py`. Individual experiment directories
contain their scripts, configurations, metrics, hashes and evaluation output.
