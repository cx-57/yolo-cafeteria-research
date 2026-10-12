# YOLO performance investigation — October 11, 2026

## Conclusion

The reported 73.8% is **mAP50**, not classification accuracy. The matching checkpoint is `model/runs/yolo11n-1x/weights/best.pt`, selected at epoch 40 for mAP50–95 of 50.3%. There is evidence of incomplete target annotations, missing class coverage, a skewed validation sample, and potential split leakage. These are higher-priority issues than an unverified model-capacity limit.

The run has not demonstrated a hard learning ceiling: the best localization metric improved at epoch 40, and logging stops at epoch 42 of a planned 80. A CSV cannot establish why training stopped. Checkpoint metadata confirms that both best and last checkpoints retain optimizer state.

## Update: manually verified fixes and stricter checks

The October 11 review branch adds 6 visible target boxes to 3 source labels. The plate sandwich/spoon scene now includes food and utensil; the second plate photo includes sandwich and salad food; the outdoor trash photo includes orange and green wrappers in addition to its preexisting bottle box. The legacy trash source stores wrapper as original ID 9 (current generated class ID 8). The metallic fork in the second plate photo is not automatically relabeled as a disposable utensil.

These corrections address **three specific images**, not 344 LVIS images or the full 3,096-image corpus. Most mixed-source images remain unverified. The previously held 728 Dry & Wet candidates also remain held; no unreviewed images are added by these code changes.

The builder now uses the historical near-duplicate candidate audit for grouping **by default** if that file is present. This is a preventive measure against evaluation leakage, not proof that every flagged pair is truly a duplicate. Both split membership and validation labels can therefore differ from the earlier ungrouped benchmark. The recorded old mAP must **not** be treated as a fixed-split control for future runs without reevaluating the old checkpoint on the rebuilt validation set.

Automated tests now cover source label repairs, malformed/misaligned model metrics, a full freshly generated real-label scan, and a cross-split audit-pair leak check. This checks data/metric integrity, not whether all objects are labeled or whether a YOLO checkpoint is accurate. The GitHub repository does not contain trained `best.pt`; real checkpoint evaluation and fine-tuning still require an available model run.

## Evidence from the saved training run

| Measure | Best mAP50–95 checkpoint, epoch 40 |
|---|---:|
| Precision | 82.768% |
| Recall | 72.083% |
| mAP50 | 73.826% |
| mAP50–95 | 50.323% |

The highest mAP50 observed was 74.013% at epoch 31. The best precision and best recall in the CSV occur at potentially different operating points/epochs; do not combine separately maximized metrics into one model's performance.

At epoch 42, mAP50 was 73.160% and mAP50–95 was 47.629%. Training box loss fell from 1.36271 to 0.92474; validation box loss fell from 1.50639 to 1.07678. This is continued optimization with a train/validation gap, not proof that more epochs cannot help. Epochs 36 and 37 are absent from the CSV. Do not silently interpolate or invent them.

The actual run used YOLO11n, 640px inputs, batch 4, MPS, and AdamW selected by `optimizer=auto`. Its effective initial learning rate was 0.000667; the nominal `args.yaml` value of 0.01 was overridden. Mosaic probability was 0.3 and was scheduled to close only for the last 10 of 80 epochs. The logged run did not reach that phase. The old 10× run has only one logged epoch and selected MuSGD at 0.01: it is not an equal-budget comparison with the 1× run.

## Data coverage and evaluation limits

There are 3,096 original photographs: 2,477 training, 619 validation, and no independent test set. The label IDs are converted only in generated training copies; originals retain their provenance IDs.

| Class | Training images with label | Validation images with label |
|---|---:|---:|
| food | 819 | 158 |
| plate | 255 | 103 |
| paper_boat | 0 | 0 |
| utensil | 153 | 40 |
| hinged_container | 78 | 43 |
| napkin | 121 | 0 |
| drink_carton | 871 | 353 |
| condiment_packet | 44 | 0 |
| wrapper | 180 | 14 |
| bottle | 300 | 58 |
| bowl | 75 | 0 |

Images can contain multiple classes, so column totals are not dataset sizes. **Only seven of eleven classes have validation targets.** Ultralytics computes AP for supported target classes; the headline mAP does not measure all eleven intended classes. Paper boats cannot be learned from this corpus. The other three unsupported validation classes cannot be assigned a defensible AP from these validation images.

The six training-only sources contribute 1,180 images. Reserving 20% of the whole corpus from the remaining eligible pool holds out about 32.3% of that pool. The deterministic selection is not stratified by class/source. Cartons appear in 353/619 validation images; wrappers appear in only 14. This skew changes support and uncertainty, but does **not** mean macro mAP weights cartons proportionally to image count.

## Confirmed annotation conflicts

1. `data/processed/plate/images/train__lvis_000000346262.jpg` contains a large sandwich and a clearly visible white disposable spoon. The label contains **only one plate box**. Neither the sandwich nor spoon is boxed.
2. `data/processed/plate/images/train__lvis_000000004386.jpg` contains unboxed sandwich and salad. A visible fork also needs review against the disposable-utensil definition. The label again contains only the plate box.
3. The plate manifest marks all 344 LVIS images provisional. The import records explicitly limit annotation to plates and do not establish complete annotation for other cafeteria targets. Other source screening records likewise distinguish structural validity from exhaustive multiclass annotation.

Ordinary detection training treats unannotated targets as background. These mixed scenes therefore conflict with food/utensil supervision from other sources. The examples confirm the labeling mechanism; they do not quantify its exact contribution to the current mAP ceiling. Complete target annotation and a controlled source ablation are needed to measure that contribution.

An American-food KFC example is in the review queue for possible incomplete coverage. Its rear fries already have a food box, so it is **not** reported as a confirmed completely missing-fries annotation. Review the overlays before changing labels. Broad source categories and manifest filenames are insufficient evidence for automatic relabeling.

## Potential leakage

Cross-checking the historical near-duplicate audit against the active split found **28 candidate pairs crossing train/validation**: MacWorX 9, Tetra Pak 15, Trash Detection 1, and Utensil 3. The corresponding historical components contain 56 images across 22 mixed components.

These are dHash/thumbnail screening candidates, not proof of identical images. The existing builder grouped original filenames and exact decoded-pixel duplicates but ignored these recorded near-components. Conservatively grouping the candidates prevents this known risk without deleting photos. Leakage tends to overstate evaluation performance; repairing it may lower the score while making the benchmark more credible.

## Revisions

`scripts/diagnose.py` now reports class image/instance support, a small-box size heuristic after letterboxing, and separately selected best metrics and missing logged epochs. It also creates a training-source exclusion experiment with **the original validation directory and labels fixed**. It reads generated labels and does not modify any original photograph or annotation. `--metadata-only` uses the saved split summary without reading cloud-backed labels, explicitly marking instance and size counts unavailable. This investigation's saved `diagnosis.json` uses that mode because some labels were macOS iCloud placeholders. It is not a fresh geometry audit.

`scripts/train.py` now exposes learning-rate, optimizer, scheduler, patience, mosaic, workers, and validation size controls. Evaluation checks checkpoint class names/IDs, reports precision/recall/F1/AP and support per class, and explicitly marks absent validation classes unsupported. The optimizer must be named when supplying an initial learning rate, because `auto` overrides it. Final evaluation uses the same requested image size as training.

The builder supports conservative grouping from an audit-pair CSV and an alternate generated output folder. The original active split is retained for comparisons. A grouped split is a **new benchmark**, so its scores cannot be compared directly to the original split's scores.

## Repeatable commands

Run diagnostics and generate the fixed-validation plate-source ablation:

```bash
python3 scripts/diagnose.py --dataset data/training \
  --run model/runs/yolo11n-1x \
  --output data/audits/model_plateau_2026-10-11 \
  --exclude-training-source plate --metadata-only
```

Recover the interrupted original training schedule without changing its settings:

```bash
python3 scripts/train.py --resume model/runs/yolo11n-1x/weights/last.pt
```

Test a short fine-tuning phase from the best checkpoint, on the **existing** split. This is a fresh optimizer/scheduler run, not a resume. The low learning rate and no-mosaic phase are a hypothesis to test, not a proven improvement:

```bash
python3 scripts/train.py data/training/data.yaml \
  --model model/runs/yolo11n-1x/weights/best.pt \
  --name yolo11n-finetune-no-mosaic --epochs 20 --batch 4 --imgsz 640 \
  --device mps --optimizer AdamW --lr0 0.0001 --lrf 0.1 \
  --cos-lr --mosaic 0 --close-mosaic 0 --patience 10 --workers 0
```

Compare the above with the same recipe using `data/audits/model_plateau_2026-10-11/ablation/data.yaml` and a distinct `--name yolo11n-finetune-no-lvis`. The prepared ablation has 2,222 training paths after removing 255 LVIS plate training images; all 619 validation images are retained. Both experiments start from the same best weights. The checkpoint has already seen LVIS plate training, so this tests removal during fine-tuning, not a clean causal estimate of LVIS exposure. It also removes most plate training examples; monitor plate AP alongside food/utensil AP. Validation labels remain provisional.

Prepare a separate split that respects the historical near-duplicate candidates:

```bash
python3 scripts/train.py --build-only \
  --group-audit data/audits/day1/near_duplicates.csv \
  --output data/training-grouped
```

For a capacity experiment, change only `--model yolo11s.pt` on a fixed, corrected split first. For a resolution experiment, change only image size. Do not bundle a bigger model, higher resolution, a new split, and multiple augmentation settings and then attribute a gain to one of them.

## What still requires data work

Fully annotate every visible configured target in mixed scenes, beginning with the plate source and food sources. Add original paper-boat examples, diverse condiment packets, bowls, and napkins from the intended cafeteria setting. Use capture-session groups for held-out validation and collect a separate fully annotated NHS test set. Current public-source validation cannot establish real NHS performance, even if mAP increases.

## References

- [Ultralytics detection performance metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics): AP/mAP interpretation and per-class diagnostics.
- [Ultralytics training configuration](https://docs.ultralytics.com/usage/cfg): explicit optimizer learning rates and mosaic scheduling.
- [Ultralytics model training tips](https://docs.ultralytics.com/guides/model-training-tips): annotation accuracy and representative data.

All run/data measurements above come from this workspace; external documentation supplies the interpretation of the metrics and configuration behavior.
