# Offline food-owner candidate — not deployed

The existing seven-class YOLO11n stays the default. Camera and hardware remain
off as requested. No checkpoint was trained or overwritten for this candidate.

The explicit profile [food-expert-candidate.json](../configs/food-expert-candidate.json)
keeps six classes with the original seven-class model and gives only `food` to
the existing frozen food/container/carton specialist. Native class IDs and file
hashes are verified before loading; source names and output ownership are
separate, so the specialist's other classes cannot duplicate the primary's.

## Fixed development comparison

Same186exposed development images, same.60confidence and.50boxIoU, no threshold
search or new annotations:

| Food recognition | Current unified | Food-owner candidate |
| --- | ---: | ---: |
| Correct food boxes | 293 | 299 |
| False food boxes | 6 | 4 |
| Missed food boxes | 50 | 44 |
| Precision | 97.99% | 98.68% |
| Recall | 85.42% | 87.17% |

Ten food targets were gained and four previously correct ones were lost. All
six other classes' predictions and scores stayed unchanged. This is a modest
aggregate development gain, not a universal improvement. Known utensil/bottle
errors remain. There are no tiny foodboxes in this validation size band, so
these numbers cannot establish better crumb recognition.

On a fixed18image sample, actual CPU inference reproduced the cached candidate
predictions. Both CPU and MPS preserved the primary's non-food outputs. Measured
MPS inference averaged.200s for the two-model candidate versus.047s for the
unified model on this run; CPU averaged.458s versus.238s. These are small,
resource-dependent inference timing samples, not cameraFPS guarantees.
**The speed cost is material, so the profile was not enabled live.**

## Offline use only for evaluation

```sh
python3 scripts/live_sorter.py --detector specialists \
  --specialist-config configs/food-expert-candidate.json \
  --image /absolute/path/to/photo.jpg --output /new/path/annotated.png
```

This command opens no camera or serial port. The normal application defaults
and saved sorter config are unchanged. Native-class validation, hash checks,
ROI/clipping safety and serial fault behavior remain in place.

Reproducible local evidence is under `runs/sorter/food-expert-offline/`:
`cached-comparison.json`, `adapter-verification.json`, `cli-smoke.png`,
`training-preflight.json`, `training-architecture-preflight.json` and
`group-overlap-audit.json`. Do not interpret the existing exposed cohorts as
sealed independent generalization tests.

## Separate training proposal

The verified original training inputs contain2501food boxes across1000food-
positive images. Only45food boxes fall below32²pixels after longest-side scaling
to640;174aremedium and2282large. The validation set contains0small,32medium,
311large foodboxes. Input labels and partition membership were only read.

[Two matched, bounded scale-augmentation trials](small-food-training-proposal.md)
are prepared. They require explicit training approval under the original user
instruction. No training or live promotion has started.
