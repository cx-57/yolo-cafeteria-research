# V5 correction candidate — November 14, 2026

The local correction candidate now runs the saved carton, wrapper, and utensil
correction checkpoints as gated recovery experts. It preserves every incumbent
prediction and adds qualifying, nonduplicate proposals. The direct checkpoint
replacements were rejected because they regressed on development data.

The inference runner previously omitted recovery experts entirely. That path now
checks checkpoint and implementation hashes, validates class ownership and rule
parameters, applies the existing wrapper geometry policy, and records the recovery
settings in its inference receipt.

At confidence 0.25 and matching IoU 0.50:

| Correction-training examples | Before | Selected candidate |
| --- | ---: | ---: |
| Cartons detected | 0 / 2 | 2 / 2 |
| Wrappers detected | 2 / 12 | 5 / 12 |
| Utensils detected | 4 / 10 | 6 / 10 |
| Wooden utensils, a subset of the above | 0 / 3 | 1 / 3 |

False positives on those respective correction groups stayed at 0 cartons,
1 wrapper, and 0 utensils. The separate, previously inspected diagnostic group
had unchanged target-class TP/FP counts. These photos are training replay and
known diagnostics, not independent test evidence. Seven wrappers and two wooden
utensils in the correction examples still remain missed.

On the complete 226-image development cohort, six-class mAP50 changed from
86.30% to 86.14%, and mAP50–95 from 59.76% to 59.69%. Each class passed the
predeclared retention gate: AP50 and AP50–95 each within two percentage points
of its incumbent. These scores do not establish independent field accuracy.

The utensil correction run completed one epoch before its disk guard stopped
training. Its direct replacement scored 53.52% AP50 / 30.43% AP50–95 and was
rejected. One fixed recovery rule, declared before its replay results, retained
the original detector and scored 68.85% / 45.57%, versus 69.55% / 45.86% before.
The custom pooling modules and frozen utensil tensors were verified preserved.
No additional training was launched in this continuation.

All assets are under:

`~/Desktop/yolo-cafeteria-research-assets/artifacts/v5-hard-example-correction-20261113/`

- `verified-correction-20261114/candidate.json`: runnable correction manifest.
- `verified-correction-20261114/verification.json`: full development scores and hashes.
- `verified-correction-20261114/end-to-end/receipt.json`: actual 15-photo pipeline run.
- `verified-correction-20261114/end-to-end-agreement.json`: predictions match the frozen composition on all 15 photos.
- `verified-correction-20261114/replay-checks.json`: separate correction and diagnostic counts.
- `verified-correction-20261114/previews/index.html`: before/after detection previews.
- `tests-20261114.log`: full suite, 161 tests run, one skipped, no failures.
- `PROGRESS_20261114.md`: execution decisions and rejected alternatives.

To infer on a new folder, use a new output directory:

```sh
TASK_ASSETS="$HOME/Desktop/yolo-cafeteria-research-assets"
"$TASK_ASSETS/venv/bin/python" scripts/v5_fresh_inference.py \
  --candidate "$TASK_ASSETS/artifacts/v5-hard-example-correction-20261113/verified-correction-20261114/candidate.json" \
  --images /path/to/photos \
  --output /path/to/new-output-directory
```

The candidate adds three detector passes to the existing five. Throughput has
not been benchmarked. The original seven-class runner and frozen checkpoints
remain separate from this six-class research candidate. Independent annotation
and evaluation of fresh school photos are still required.
