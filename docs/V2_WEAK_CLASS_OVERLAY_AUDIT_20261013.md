# October 13 V2 weak-class prediction review (38 original validation images)

## Evidence / reproducibility
The user supplied `v2-weak-fast10-review.zip` (SHA-256 `02b39e4e9b038ad356a87920946ab7d404f43e019eaf14c5f0140f84238b3d76`), containing 38 side-by-side source/YOLO prediction overlays, `summary.json` and 99 queued unmatched GT/prediction rows in `review_queue.csv`. The ZIP and trained `best.pt` live outside this repository. This report is derived from the actual overlays/CSV, not inferred from logs or an unperformed model run.

Inference: `v2-weak-fast10/weights/best.pt`, 640-pixel inference, confidence **0.20**, matching IoU **0.50**. The 38 images were **selected**, not a random benchmark sample: 15 LVIS plate-source images and 23 Trash Detection images, 591 total images in the underlying unchanged public-source V2 validation split. Image names containing `train__` or `valid__` retain **original source-split identifiers**; every image in this ZIP was drawn from the new `val` directory. No independent NHS cafeteria test was run.

## Measured errors (selected 38-image diagnostic ONLY)

| Target | Ground-truth objects | Same-class matches | Selected-cohort recall |
| --- | ---: | ---: | ---: |
| Plate | 32 | 16 | 50.0% |
| Wrapper | 9 | 1 | 11.1% |
| Bottle | 26 | 13 | 50.0% |

The 38-image diagnostic also includes 27 labeled food objects (13 matched). The selection mechanism is concentrated on difficult objects. These counts **do not replace** Ultralytics AP50 or its per-class `recall` (which uses its own metric operating point). The initial viewer used greedy IoU matching, which can undercount crowded plates; the later viewer code uses maximum-cardinality same-class matching. These original observed numbers have **not** been silently recalculated or modified.

### Object area is a strong failure mode

- **Plates occupying <1% of frame:** 3/12 matched. **Plates >=16% of frame:** 6/7 matched.
- **Bottles occupying <1%:** 3/15 matched. **Bottles 1–4%:** 8/8 matched.
- Source scenes include tiny edge-clipped/background plates, large/occluded food over plates, and bottles lost among vegetation or trash. Larger input alone and naive sliced inference had already failed separate diagnostic tests. New in-context small-object **training coverage** is a stronger next hypothesis than another untuned image-size increase.

### Directly observed wrong-class overlaps (not label certification)

Comparison of the uploaded *unmatched-label* and *unmatched-prediction* box coordinates finds:

| Source image (short) | Actual labeled class | Predicted class | Box IoU | Predicted conf |
| --- | --- | --- | ---: | ---: |
| `trash_detection__train__000055_jpg.rf.29daca…` | wrapper | bottle | **0.883** | 0.211 |
| `trash_detection__train__IMG_5041_JPG.rf.901be…` | wrapper | condiment_packet | **0.744** | 0.362 |
| `plate__train__lvis_000000274804` | food | plate | 0.543 | 0.545 |
| `plate__train__lvis_000000515662` | plate | food | 0.541 | 0.240 |

The first two are visually supported wrapper-vs-other-class confusions: flattened snack packet on outdoor ground, and a Lay's snack bag. Plate/food overlap often occurs because both physical objects occupy the same pixels, so the other two are *review candidates* rather than certified misclassifications.

### Domain mismatch is visible, not merely inferred

- Trash Detection images show packaging and bottles on **roadsides, in gravel, on grass, beside fences and buried in litter** rather than resting on cafeteria trays. Example: `trash_detection__train__000085_JPG.rf.d198…` has seven missed tiny bottle boxes amid plants and discarded items.
- LVIS plate scenes include people dining far away, nearly hidden plate rims under food, a tiny distant plate in a livestock/stable setting, and a cat/pastry scene with only a thin visible plate edge.
- `trash_detection__valid__000082_JPG.rf.7a7…`: small crumpled outdoor wrapper, no detection at 0.20.
- `plate__train__lvis_000000022411`: distant crowded dining scene with eight unmatched plate labels and five unmatched plate predictions.

Do **not** auto-delete such challenging samples or assume their original boxes are wrong. They expose the current model's weaknesses, but public-source AP is not NHS cafeteria AP.

## Actions / constraints

1. **Hold training now.** The last 10-epoch weak-class fine-tune changed the classification/recall trade-off without achieving strong AP: wrapper AP50 0.298, plate AP50 0.444, bottle AP50 0.477; overall AP50 0.736.
2. **Add source-domain matched, diverse still photos**: cafeteria trays, wrapper/bottle side-by-side hard negatives, crushed snack packaging, partially food-covered plates, realistic object scale and varying lighting/backgrounds. Use separate capture sessions/days and *all 11* target labels on mixed scenes. More repetitions of the same outdoor litter will not substitute for this.
3. **Visually adjudicate the highest-priority images** with the original full-resolution labels before accepting any box correction; only add audited changes to V2 copies with source hashes. Keep immutable V1 labels and frozen validation membership. Do not relabel wrappers as bottles based on the model's opinion.
4. **Keep exact independent NHS scenes for the final held-out test.** V2 lacks positive validation for paper_boat, napkin, condiment_packet; bowl has only two objects, insufficient to claim reliability.
5. **Retain matched controls** for future experiments (same images, training budget, seed, augmentation, validation). Candidate source-aware loss is a hypothesis, not a demonstrated fix. The current Mac reported Ultralytics 8.4.80 whereas repo pins 8.4.175; align the environment before API-dependent loss comparisons.

## Engineering fix to diagnostics

`scripts/inspect_predictions.py` now does maximum-cardinality distinct same-class matches instead of greedy pair selection (avoids undercounting overlapping plates). It also writes `wrong_class_candidates.csv` recording candidate high-IoU overlaps, explicitly marked as **potential confusions** rather than adjudicated relabeling. These changes improve error accounting; **they have no effect on trained weights or official Ultralytics AP**.

**Next evidence needed:** verified NHS in-domain training/validation images and rerun of actual predictions on an independently labeled cohort. Do not assert a new percentage improvement until measured.

## October 13 remediation implementation (post-audit)

The following changes address measurable **software/data-integrity** issues. They do
**not** certify that weak-class recognition is fixed, because no new 80%-recall
checkpoint or independent NHS test exists.

| Issue | Change | Verification |
| --- | --- | --- |
| Image review undercounts crowded matching | Maximum-cardinality 1:1 IoU matching in `inspect_predictions.py` | Unit tests and CI |
| Wrapper confused with bottle/condiment | New `wrong_class_candidates.csv` from overlapping unmatched GT/predicted classes | Unit tests; must visually adjudicate candidate errors |
| Only 38 selected images reviewed | `--all-images --no-overlays` runs 591-image error scan with compact CSV | Unit test; run on user's actual trained checkpoint |
| Misleading bowl/wrapper 99% headline | Class support gate: AP for zero objects undefined, below 30 = insufficient support | `metrics.json` includes `coverage.insufficient_support_classes` |
| Ultralytics 8.4.80 on user's Mac vs tested 8.4.175 | Fresh training refuses mismatched installed version unless `--allow-unpinned` | Unit test |
| Weak-class training diversity | Eleven Commons images (31 **provisional** 11-class boxes); combined with five pre-existing paper-boat images | Train-only optional V2 builder + CI, license/artist metadata preserved |
| Human-readable candidate review | CI exports an eleven-photo labeled contact sheet to facilitate human checking | Pipeline artifact |
| Val missing paper_boat/napkin/condiment_packet; bowl support 2 | Still open. Cannot add unverified photo/boxes to validation | Source-disjoint NHS or fully reviewed external val is needed |
| Real cafeteria generalization | Still open. No independent in-domain NHS held-out images or evaluation | Capture labeled images from separate sessions/days |

### Run a COMPLETE error scan on the existing checkpoint (no retraining)

```bash
git pull
python3 scripts/inspect_predictions.py \
  --weights model/runs/v2-weak-fast10/weights/best.pt \
  --data data/training-v2-weak-focus/data.yaml \
  --split val --all-images --no-overlays \
  --conf 0.20 --iou 0.50 --device mps \
  --output model/v2-weak-fast10-full591
```

This creates `summary.json`, `review_queue.csv`, and
`wrong_class_candidates.csv`; the CSV rows are **review candidates**, not
certified false positives or verified errors. AP is computed separately by
Ultralytics, and the full split is still public-source development validation,
not a final NHS benchmark.

### Optional NEW train-only exploratory data (unverified)

The `data/v2_candidates/provisional_train_20261013.csv` contains 16 distinct
candidate photos: five earlier LVIS paper-boat draft images with seven
provisional target boxes, plus eleven Wikimedia Commons photos with 31 proposed
boxes for plates/food, wrappers, bottles, napkins, packets, bowls and utensils.
File page/licensing/artist metadata are under
`data/v2_candidates/weak_cafeteria_20261013/SOURCES.json`.

```bash
python3 scripts/v2.py build \
  --source data/training \
  --output data/training-v2-weak-exploratory-20261013 \
  --provisional --use-codex-review \
  --candidate-manifest data/v2_candidates/provisional_train_20261013.csv
```

**Do not treat these provisional AI-assisted annotations as human-verified.**
Review the CI contact sheet and **all original full-resolution photos and all
11 target classes** before using them for publication-quality experiments.
Do not put any candidate in `val` or `test`; the build enforces this.
Existing 2,505 original train / 591 val copies and all historical checkpoints
remain untouched. The new experimental build should have **2,521 train and
591 val** images. Modest candidate additions cannot establish generalization.

### Collection and stop rules for actually resolving the open model failures

Collect real **still photos, not video**: discarded or uneaten food packaging on
NHS tables/trays, empty and food-covered plates, clear/crushed drink bottles,
condiment sachets versus snack wrappers side by side, napkins, paper boats,
bowls and clamshells. Vary scale, occlusion, clutter, lighting and days.
Exclude identifiable bystanders; keep source/permission and day/session metadata.

Use session-disjoint splits: at minimum one set of *complete* 11-class
annotations for training, one separate group/day for development validation,
and another unseen group/day for the final test. Aim for **at least 30 verified
objects of EACH of the 11 classes in validation**, with many more for stable
performance estimates (30 is a reporting gate, not a statistical guarantee).
Do not move 591 legacy public validation photos into training as a quick fix,
or silently rename public images as independent NHS data.

Acceptance requires **measured** wrapper, bottle and plate precision/recall/AP
on the same valid split, no regressions to food/carton/container, plus a separate
NHS evaluation; the target of >=80% class recall/mAP50 is an engineering goal,
not a measured fact. A code commit, new candidate image, or 10-epoch training
log alone cannot close a model-performance issue.

### Reproducibility safeguard

```bash
python3 -m pip install -r requirements.txt
python3 -c 'import ultralytics; print(ultralytics.__version__)'
```

Expected `8.4.175`; your earlier `8.4.80` model metrics should retain their
original version provenance. Never rewrite past metrics or imply historical
model output was produced with a later pinned library.
