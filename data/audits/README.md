# Dataset audits

The current model has **11 classes** (see `config.yaml`). Run `python3 scripts/audit.py --manifests data/manifests --output data/audits/current` to regenerate the audit after changing the data. The old audit lives at `data/audits/old_11class_snapshot/` and documents the historical labels before TartanHacks images were included.

The [October 11 performance investigation](model_plateau_2026-10-11/README.md)
explains the 73.8% mAP50 run and provides an annotation review queue, split
leakage evidence, and controlled experiment commands.

**Source-level screens are not proof of complete annotations.** Examples can have unannotated target objects; correlated images/video frames can inflate validation scores. Preserve capture groups and hold out independent natural NHS scenes for final testing.

## October 12: 100-epoch checkpoint investigation (source-level, NOT inference)

User-supplied console output from YOLO11n at 100 epochs: 619 validation images,
1,472 annotations, **0.779 mAP50**, **0.573 mAP50–95**. Per-class figures:
food 0.991; plate 0.559; utensil 0.851; hinged container 0.995;
drink carton 0.977; wrapper 0.435; bottle 0.642.
The following classes are **not evaluated**: paper boat, napkin,
condiment packet, and bowl. No independent NHS test has been performed.

### Diagnosed validation-source mismatch

- **Plates**: the 103 plate validation photos belong to the LVIS plate source.
  Seen examples include tiny/edge-clipped plates, plates obscured by food,
  and multiple partially overlapping physical plates. The 344 LVIS source
  images remain tagged `provisional` and were not exhaustively reviewed
  for all 11 target classes. The 14 Waste Detection v3 plate photos are
  training-only. An LVIS validation score is not an NHS plate score.
- **Wrappers**: only 15 validation images / 19 target instances. The only
  validation-eligible wrapper source is Trash Detection; manually inspected
  examples show outdoor litter, not cafeteria tray scenes. A score based
  on 19 targets has high uncertainty. Precision 0.873 and recall 0.363
  suggest misses at the validator's operating point; a prediction review is
  necessary to separate genuine misses from incomplete ground-truth boxes.
- **Bottles**: 59 validation images / 76 instances, likewise drawn from
  Trash Detection. Checked examples include bottles lying on gravel, beside
  roads and in grass. Other newly added bottle photos are training-only.
  Precision 0.751 and recall 0.539; whether background, size or material
  causes errors is not determined without checkpoint inference.
- **Napkins**: 121 labeled training images and **zero validation**.
  Manifest counts: Waste Detection v3 **52**, TartanHacks **35** and
  Dry & Wet Waste **34**. The first two include correlated video frames.
  Treat the 121 as *images*, not 121 independent capture sessions.
- **Condiments**: 44 photos, all tagged
  `condiments_same_background_batch` in the provenance manifest.
  No held-out condiment evaluation; high overfitting risk.
- **Bowls**: 75 Waste Detection v3 photos, training-only, including many
  frames from a small number of source videos; no validation AP.
- **Paper boats**: zero labeled train and validation photos.


### Additional direct visual observations (not yet relabeled)

- `data/processed/plate/images/train__lvis_000000225490.jpg`: multiple pizza
  slices visibly occupy one of two annotated plates. The stored label file has
  two **plate** boxes and no **food** boxes. This is another confirmed example
  of incomplete multi-class supervision; deciding whether to box each slice or
  the entire portion requires a consistent food annotation policy.
- `data/processed/plate/images/train__lvis_000000478368.jpg`: seven visible
  pizza-topped muffin pieces sit on a baking tray. The label contains a tiny
  **plate** box at the upper-left edge and no food annotations. This scene
  strongly differs from a conventional cafeteria tray/plate photograph.
- `data/processed/plate/images/val__lvis_000000557582.jpg`: two visible pizza
  pieces are being held by people. The sole annotated plate is a thin partially
  visible strip near the bottom. This image exemplifies severe occlusion and
  incomplete food annotation **inside the validation set**.
- `data/processed/plate/images/train__lvis_000000001455.jpg`: four plate
  boxes overlap within a small stack of paper plates. Overlap may be physically
  valid because plates are stacked; the geometry is a **review candidate**, not
  proof of duplicate annotations.

A missing ground-truth food box can cause a correct food prediction to be
miscounted as a false positive. Fixing such validation labels changes the
benchmark: do not compare old and new AP as though the dataset was identical.

### Required next measurements

1. Analyze **actual best.pt predictions** on the frozen grouped validation set.
   Export per-image unmatched predictions and false negatives at explicitly
   declared confidence/IoU thresholds, plus native Ultralytics per-class AP,
   PR curves, confusion matrix and visual overlays. Flag suspected missing
   ground-truth annotations separately from confirmed model false positives.
2. Visually review highest-error images, particularly LVIS plates and Trash
   Detection wrappers/bottles. Do not auto-relabel from predictions.
3. Before changing the trained model, add complete capture-group-disjoint
   validation for napkins, bowls and condiments; source images must be
   re-reviewed for exhaustive 11-class annotation. Acquire genuinely new
   paper-boat samples. Validation membership changes require retraining
   experimental baselines from the same pretrained checkpoint.
4. Make a separately held-out, fully labeled NHS cafeteria test set with
   distinct days/capture sessions. Avoid tuning on that test.
5. After data corrections, compare one change at a time (YOLO11n versus
   YOLO11s, image size, source ablations). Record source-wise and per-class
   AP50 and AP50–95, precision/recall and bootstrap uncertainty as
   appropriate. Do **not** claim 95% achieved without independent evidence.

### Limitations / verification status

The repository does **not** include the trained `best.pt` or this run's
`results.csv`/prediction overlays. The new console output establishes
reported aggregate metrics, **not** which specific objects were missed.
This is a provenance and sampled-image audit rather than a complete
frame-by-frame relabeling or end-to-end prediction error analysis.
The existing CI checks label structure and audited duplicate grouping;
it cannot prove semantic completeness or model accuracy.
