# Utensil / condiment-packet preflight — 2026-10-20

Target: **85% AP50 per target class**, using YOLO11n. AP50 measures detection, not image-classification accuracy. **The target is not met. No accuracy gain is claimed for the new V7 supervision until training and saved-checkpoint evaluation finish.**

## Latest measured performance

The safer V6 epoch-16 saved checkpoint (`recovery80-joint-v6/train/weights/best_utensil.pt`, SHA256 `5cf0eca1767d703b1172ffaf616e8f8d9d43c98e93863946bcd61f122e516f48`) was reloaded again for validation on the unchanged V7 validation set:

| Class | Precision | Recall | AP50 | AP50–95 | Validation objects |
|---|---:|---:|---:|---:|---:|
| Utensil | 30.89% | 20.61% | 20.51% | 7.60% | 296 |
| Condiment packet | 3.89% | 11.90% | 2.44% | 1.20% | 42 |

Latest test: `preflight/utensil-packet-test-v7.json` (764 validation images; same saved checkpoint). The earlier evaluation in `recovery80-joint-v6/evaluation.json` measured 20.55% / 2.47%; this repeat measured 20.51% / 2.44%. Neither evaluates a model trained with the V7 repairs. The inherited epoch-15 checkpoint scored 22.48% / 1.83% AP50 on the same V6 benchmark. Earlier V4 scores used different packet validation support and must not be substituted for these.

The nominal-batch-4 experiment collapsed to 0.06% utensil / 0.18% packet AP50 and ran out of disk during native checkpoint saving. Its intact last checkpoint was separately evaluated; the branch is on quality hold. **Do not resume `recovery80-joint-v6-step4`.** Increasing update frequency at the inherited learning rate did not solve the problem.

Live validation and a saved-checkpoint evaluation previously differed (17.07% versus 20.55% utensil AP50); separate batch/fusion diagnostics did not explain that discrepancy. Saved-checkpoint evaluation remains canonical. The new full run cannot stop merely because a provisional live AP reaches the target.

## Confirmed data/supervision problems and V7 repairs

The source policy supervised packet negatives only inside packet-positive photos: **zero packet-free active training photos provided negative BCE for the packet class**. At confidence 0.25 / IoU0.5, the epoch-16 model found all 44 old same-background training packets, but also produced high-confidence packet boxes on cartons, chip bags and cutlery. These are fixed-threshold diagnostics, not AP50 or proof that every unmatched detection is false. Evidence: `preflight/source-diagnostics.json` and visually inspected `preflight/unmatched-class7.jpg`.

V7 preserves all 3,145 images and their split/hash/original/capture identities. **Every one of the 764 validation images and label files is unchanged from V6**, so the final V7 metrics can be compared with the V6 baseline at the same inference settings. Only training supervision changes:

- Add 18 previously unboxed eating spoons, using whole-image review and native coordinate overlays.
- Correct one training tray: remove a chopstick box and a handle-only spoon box; add one complete eating-spoon box. Chopsticks remain outside this project's eating fork/spoon/table-knife definition.
- Add a visible clipped eating fork and knife to an existing training packet photo. Its sauce box is unchanged.
- Record 229 hash-bound training-photo reviews. These add **152 verified packet-free photos and 102 additional verified utensil-free photos** to negative supervision. Ambiguous sachet-like tray items retain unknown packet coverage.
- Keep the original shared partial-label implementation untouched. The recovery trainer adds reviewed image/class coverage to its existing mask; every assigned positive remains supervised, and unreviewed classes retain their previous policy.

| Class | Active positive training photos | Training objects | Validation positive photos | Validation objects |
|---|---:|---:|---:|---:|
| Utensil | 167 | 265 | 230 | 296 |
| Condiment packet | 74 | 77 | 22 | 42 |

The full pool is 2,381 train / 764 validation. The active list still has the **same 613 training filenames as V6**, including all 319 currently eligible utensil/napkin/packet-positive photos and 294 other-class replay photos. All known target positives are retained; the remainder of the corpus is not assumed to be target-free. One previously identified plastic-bag/napkin error remains held out of active training.

V7 dataset: `data/utensil-focus/grouped-v7-reviewed-negatives`, fingerprint `cc30ea3b8886e87caf6252f9ee62dd5cdd3814bbf774072a4d7beb224aa3a1bd`.

Review/provenance: `data/utensil-focus/training-reviews-v7.json`, `packet-additions-v7.json`, `target-coverage-reviews-v7.json`, and `coverage-reviewed-v7.json`. The 49 existing Commons packet photos retain their per-image author/license/URL records and photographer grouping; no new online photos were added in this preflight.

Machine-readable before/after evidence: `preflight/data-readiness-v7.json`. Geometry, membership, original/byte/pixel/capture leakage, source hashes, and all photo-level coverage proofs are checked before training. Unknown related captures remain possible.

## Verification

`preflight/native-loss-smoke.json` records an actual batch 4 MPS forward/backward pass. Packet-negative gradients on a verified packet-free carton and tray changed from `[0,0]` under the previous policy to `[7.49647,1.82642]` under V7; all model gradients were finite. **No optimizer steps, training epoch, checkpoint write, or accuracy improvement occurred in that smoke test.**

The full repository suite passes 215 tests (`preflight/tests-full-suite.log`). Independent code/data review found no important issues. The original 11-class source data/pipeline and the separate plate/wrapper specialist are preserved. Protected specialist SHA256: `f7055a6bddfaa2e84480afd7c875571333f7cb4e30b941236eb57857df85d8f1`.

## Completed repaired-data pilot

The disk-space cutoff was removed at the user’s request. A fresh V7 pilot completed **one training epoch**, saved optimizer-bearing checkpoints, rechecked dataset integrity, and separately evaluated the saved selection on all 764 unchanged validation images.

| Class | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
| utensil | 70.35% | 1.69% | 15.10% | 4.50% |
| condiment_packet | 0.90% | 7.14% | 0.64% | 0.15% |

Evidence: `recovery80-reviewed-v7-pilot/evaluation.json`. These scores are poor; the 85% target is not met. This fresh epoch 1 result is not a matched-duration comparison with the older V6 epoch 16 model and does not establish a data-repair accuracy gain. The pilot verifies actual training, checkpoint saving, reload evaluation, and full optimizer preservation.

To continue this tested run through **80 total epochs** (79 remaining), use:

```bash
caffeinate -i .venv/bin/python -u scripts/utensil_recovery.py train --resume \
  --data data/utensil-focus/grouped-v7-reviewed-negatives \
  --training-yaml data/utensil-focus/weak-v7-reviewed/data.yaml \
  --policy data/utensil-focus/coverage-reviewed-v7.json \
  --targets utensil condiment_packet \
  --validation-batch 4 --full-schedule \
  --model-dir model/utensil-focus/recovery80-reviewed-v7-pilot
```

Do not pass a minute budget for the overnight continuation. The saved checkpoint retains the original 80-epoch horizon, optimizer and EMA. The alternative fresh-run command below remains available but restarts from epoch 1 in a separate directory.

## Tonight: fresh 80-epoch run

The disk-space cutoff was removed at the user’s request. Training does not enforce a minimum free-space threshold.

This is a fresh run from `model/weights/yolo11n.pt`, with 80 total epochs, actual batch 4, nominal batch 64, 640px, MPS, AdamW base LR 0.001 and validation batch 4. It retains the 11-class ID layout; the two target classes drive checkpoint selection. Existing photo/geometric augmentation remains; mixed-image augmentation stays disabled because it would invalidate the per-image coverage policy. Full-schedule mode disables native patience and provisional target-score stopping, while retaining the checkpoint with the best minimum target AP50.

The dataset and training list already exist. Verify them with:

```bash
.venv/bin/python scripts/utensil_recovery.py verify \
  --data data/utensil-focus/grouped-v7-reviewed-negatives \
  --training-yaml data/utensil-focus/weak-v7-reviewed/data.yaml \
  --policy data/utensil-focus/coverage-reviewed-v7.json
```

Run from the repository root. `caffeinate` prevents idle sleep while the process runs:

```bash
caffeinate -i .venv/bin/python -u scripts/utensil_recovery.py train \
  --data data/utensil-focus/grouped-v7-reviewed-negatives \
  --training-yaml data/utensil-focus/weak-v7-reviewed/data.yaml \
  --policy data/utensil-focus/coverage-reviewed-v7.json \
  --targets utensil condiment_packet \
  --validation-batch 4 --full-schedule \
  --model-dir model/utensil-focus/recovery80-reviewed-v7
```

Do not add `--continue-from`, `--expand`, `--nominal-batch 4`, or a minute budget to this launch. A later V7 attempt left startup plots/arguments but no completed epoch or checkpoint. Those files are preserved in `recovery80-reviewed-v7-startup-only`; the documented fresh-run output path is available again. The previous optimizer state is not reused with this supervision change. If this new run is interrupted after a completed epoch, repeat the same command with `--resume`; keep every dataset/policy/full-schedule flag identical.

After epoch 80, the script verifies the data again, reloads the selected saved checkpoint, evaluates it and writes `model/utensil-focus/recovery80-reviewed-v7/evaluation.json`, including per-class P/R/AP50/AP50–95, overall metrics, checkpoint SHA and dataset fingerprint. Native training plots/CSV are under `train/`.

Manual reevaluation after training:

```bash
.venv/bin/python scripts/train.py data/utensil-focus/weak-v7-reviewed/data.yaml \
  --evaluate model/utensil-focus/recovery80-reviewed-v7/train/weights/best_utensil.pt \
  --device mps --imgsz 640 --batch 4 --workers 0 --name utensil-packet-reviewed-v7-final
```

Remaining limits: only 77 packet training objects, with 44 photos from one background batch; inherited labels outside targeted reviews remain provisional; validation has only 42 packet objects; and there is no untouched cafeteria-session test. Public pretraining overlap is not excluded. Completing 80 epochs cannot guarantee 85% per class or cafeteria accuracy.
