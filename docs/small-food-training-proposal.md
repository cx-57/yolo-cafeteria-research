# Bounded small-food experiment — shortened at user request

The user approved both bounded trials, then asked to shorten the run. The control completed two epochs; the scale candidate was interrupted during its first epoch before a checkpoint was saved. No further training is running. Camera and servo operation stay off. The original
checkpoint and data remain unchanged. This proposal targets scale robustness
for the existing food-serving labels; it does not silently redefine incidental
crumbs as food servings or promise orange/crumb generalization.

## Original fixed experiment — interrupted

Use the existing custom seven-class YOLO11n initialization:
`../yolo-cafeteria-research-assets/artifacts/cumulative20-100views-20261031/train/weights/best.pt`,
SHA256 `19e1294484efe9bbabbc71866ccd57f59942899cc385937dc7aeed55fc373795`.

The original plan specified two matched trials, at most two epochs each, same initialization,
seed42, existing train/validation membership, CPU, batch1, learningrate.00005,
and gradient-freeze of the first10layers. Use all2372existing training images,
not the user's few camera examples. Profile files differ in only one setting:

- [Control](../configs/train-small-food-control.yaml): original scale augmentation .15.
- [Candidate](../configs/train-small-food-candidate.yaml): scale augmentation .50.

No mosaic, no new food-specific confidence tuning, no new labels, no repeat-until-
passing trials, no held-out-test scoring or automatic live promotion. Original
input images/labels are copied by the existing training runner so native caches
cannot modify the originals. Source hashes are checked before/after.

## Review and decision rules

Compare the **last checkpoint at the same two-epoch cap** for both runs against
the unchanged parent on the fixed exposed186-image development cohort. Report
all three results, not just the winner. Keep the existing .60confidence / .50box
IoU operating point, per-class AP and class-error counts fixed. A proposed
promotion must retain foodprecision/recall and show no decline in the other-six
per-class AP50/AP50–95 against the frozenparent (floating-point tolerance only).
Any regression is reported and prevents automatic replacement. The fixed-ROI synthetic scale check is diagnostic only and cannot
be the sole basis for a claim of better small-food accuracy.

The186-image cohort has no foodboxes under32²pixels when scaled to640. Diverse,
separate physical small-food capture groups are still required before a real
small-piece performance claim. The current camera failures are development
examples, not independent validation. The test partition is preserved and not
used for candidate/threshold selection.

## Previously authorized commands — four total epochs maximum

These describe the original bounded plan. The user subsequently shortened the run; do not resume the interrupted trial automatically.

```sh
python3 scripts/training.py train \
  --data ../yolo-cafeteria-research-assets/artifacts/cumulative20-100views-20261031/train/resolved-data.yaml \
  --model ../yolo-cafeteria-research-assets/artifacts/cumulative20-100views-20261031/train/weights/best.pt \
  --profile configs/train-small-food-control.yaml \
  --output ../yolo-cafeteria-research-assets/artifacts/small-food-scale-control-20261115 \
  --device cpu

python3 scripts/training.py train \
  --data ../yolo-cafeteria-research-assets/artifacts/cumulative20-100views-20261031/train/resolved-data.yaml \
  --model ../yolo-cafeteria-research-assets/artifacts/cumulative20-100views-20261031/train/weights/best.pt \
  --profile configs/train-small-food-candidate.yaml \
  --output ../yolo-cafeteria-research-assets/artifacts/small-food-scale-candidate-20261115 \
  --device cpu
```

Read-only dataset verification completed:2372train,186val and248test images;
326,831,601bytes required for a full protected snapshot. Exact-byte and recorded
capture-group intersections are zero across splits. These inherited cohorts
remain exposed development data, not independent field tests. The checkpoint
also passed a strict499-tensor architecture roundtrip with no optimizer step.
Disk-space and source-hash checks must be repeated immediately before running. The
existing runner refuses existing output directories. Never overwrite the
frozen source or change the live sorter config as part of these commands.

[Completed shortened-run results](bounded-small-food-results.md). No training is queued.
