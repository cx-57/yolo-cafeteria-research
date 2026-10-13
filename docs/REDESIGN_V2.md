# YOLO Cafeteria V2 — Research-backed redesign (October 12, 2026)

**Status:** Proposed experimental protocol. No V2 accuracy gain has yet been measured.

## Objective

Detect 11 actual cafeteria object categories reliably, including small/crushed objects and difficult packaging, on independent NHS still photographs. Retain the existing V1 checkpoint and labels unmodified as historical baseline. Version V2 independently.

## 1. Class definitions / annotation policy

Current IDs must stay unchanged for checkpoints using the existing ontology.

| ID | Class | Positive examples and exclusions |
|---:|---|---|
| 0 | food | Edible food portions; do not duplicate one pizza slice as both a portion and every topping. |
| 1 | plate | Physical serving plate; prioritize NHS-relevant disposable paper plates, keep edge-clipped instances when identifiable. |
| 2 | paper_boat | Paperboard food boat, zero labeled V1 train examples; requires data. |
| 3 | utensil | Actual eating fork, spoon, spork or table knife, including disposable and reusable cafeteria cutlery. Exclude cooking tools and disconnected object parts. |
| 4 | hinged_container | Clamshell/takeout container, not a flexible food wrapper. |
| 5 | napkin | Cafeteria disposable napkin, not cloth napkin or random tissue. |
| 6 | drink_carton | Beverage carton, including small cafeteria milk/juice cartons and larger cartons. Exclude meal boxes, plastic bottles and flexible snack pouches. |
| 7 | condiment_packet | Individual ketchup/mustard/sauce sachet; do not label it simultaneously as wrapper. |
| 8 | wrapper | Flexible food packaging (chip bags, candy wrappers, crumpled foil/plastic snack pouches and identifiable sandwich/bread wrapping). Draw one box around the package, not its printed artwork. Generic storage bags without identifiable food use remain ambiguous. |
| 9 | bottle | Rigid/semi-rigid beverage bottle, including visibly crushed bottles. Do not label flattened snack pouches as bottles. |
| 10 | bowl | Serving bowl, including disposable cafeteria variants when in scope. |

Flag ambiguous items for adjudication; do not auto-fix labels from model predictions. Use consistent food **portion** boundaries and review the 60 nested LVIS ingredient candidates. Save original images, original labels, correction provenance, and review decisions.

V2 clarification during the complete-corpus review: actual reusable eating cutlery and larger beverage cartons stay in scope. This keeps existing useful examples with a consistent physical-object definition. V1 labels and its historical benchmark are unchanged. Printed menu pictures, clothing artwork, toy food, drink cups, cloth napkins and cooking equipment are not target instances. Identifiable clipped, stacked and food-covered plates remain positive examples.

## 2. Established diagnostic evidence

- Original V1 reported 0.779 mAP50 / 0.573 mAP50–95 on 619 source-based validation photos; only seven of 11 classes have validation targets, no NHS test set.
- 55-image experiment: full 640 detected 45/90 targets; full 960 detected 44/90; full 640 plus crops and NMS detected 53/90 but created 92 unmatched predictions vs 31 for full 640. Unmatched does **not** mean confirmed false positive when annotations are incomplete.
- Crop settings selected on 31 development images failed to improve 24-image diagnostic holdout recall (17/29 vs 17/29 for full 640, with 14 vs 10 unmatched detections).
- Direct reviewed photographs show bottle predictions on objects annotated as flexible wrappers at high IoU (~0.92, ~0.93). Prioritize class confusion and ambiguous packaging.
- V1: paper_boat 0 train photos; condiment_packet 44 photos from same-background batch; napkin training highly correlated; wrappers primarily outdoor litter scenes. Current validation has no positive examples of paper_boat/napkin/condiment_packet/bowl.
- LVIS overlay: 1,364 proposed food boxes, including 856 below 1% image area; 60 contained-ingredient candidates in seven images await review.

## 3. Data-first restart

1. **Freeze class policy** above. Review and correct missing, wrong-class and overlapping boxes, beginning with wrappers/bottles and plate-food combinations. Mark images with unknown class coverage; do not assume imports fully annotate all configured classes.
2. **Collect actual NHS still photos** with permission: trays, tables, realistic distance, varied lighting, cafeteria backgrounds, empty/full/crushed packaging, object overlap and hard negatives. Avoid recognizable bystanders. Collect varied sessions/days, not near-identical burst frames. No videos required.
3. **Sample diversity over repetition.** Aim for hundreds of independently photographed reviewed examples of each weak class, especially wrappers, bottles, paper boats, napkins, condiments, bowls and NHS plates. Explicitly photograph wrappers and bottles **on the same backgrounds** and at the same scales to teach their difference. A few hundred diverse examples can be more useful than many correlated frames; exact counts are planning targets, not guaranteed minimums.
4. **Build V2 validation and final test** from separate capture groups/days. A feasible initial target is 100–200 fully annotated development validation photos and 100–200 genuinely independent NHS test photos, with adequate object support for every class; expand until each class has meaningful positive instances. Do not select hyperparameters on the final test.
5. **Human-in-loop label review.** Prioritize high-confidence cross-class overlaps, suspicious missing labels and poor box geometry. Use out-of-sample model predictions; Cleanlab can rank candidates. Grounding DINO can propose additional candidates for human review, not supply unverified truth.
6. **Prevent leakage.** Keep source sessions, near-duplicates, original/augmented exports and video capture groups together. Record split hashes and label-policy versions.

## 4. Minimal controlled training comparisons

Do not retrain until a verified V2 data.yaml exists. The path below is a **placeholder**, not a presently available dataset. Starting learning-rate values are controlled experiment settings, not tuned optima.

**A. V2 corrected-data baseline — pretrained YOLO11n:**

```bash
python3 scripts/train.py data/training-v2/data.yaml --model yolo11n.pt \
  --name v2-control-n --epochs 80 --batch 4 --imgsz 640 --device mps \
  --optimizer AdamW --lr0 0.0007 --lrf 0.1 --cos-lr \
  --mosaic 0.3 --close-mosaic 10 --workers 0
```

**B. Built-in rare-class weighting:** rerun the exact same command with only `--name v2-weighted-n --cls-pw 0.25`. This option is supported in pinned Ultralytics 8.4.175 and was newly exposed in scripts/train.py. Verify per-class AP and regressions, particularly common food/carton classes. Class weighting cannot create missing paper-boat training images.

**C. Bigger model:** rerun A changing only `--model yolo11s.pt --name v2-control-s`. Official YOLO11 COCO benchmarks favor YOLO11s in AP (47.0 vs 39.5 mAP50–95), but it uses 9.4M vs 2.6M parameters. This is **not a predicted NHS gain**; test real MPS latency/memory.

**D. Source-aware partial labels (research ablation):** compare YOLO11n standard and existing experimental `--coverage-policy data/audits/model_revisions_2026-10-12/source_coverage.json`, with mosaic turned OFF in BOTH runs. Do not directly compare a no-mosaic trial against a mosaic control or assume policy metadata establishes full annotation completeness. Smoke-test for two epochs before a longer run.

**E. Frozen backbone (lower priority):** optionally compare `--freeze 10` for a short first stage, then unfreeze at lower LR, against a schedule-matched nonfrozen control. Do not treat data-limited transfer-learning advice as a guaranteed gain.

Before copy-paste: Ultralytics native copy_paste applies to tasks with polygon labels (segment/OBB); don't expect bbox-only detect training to perform the published mask-based Copy-Paste augmentation. Mosaic may improve scale variation; compare fixed setups rather than blindly increasing every augmentation. SAHI/crops remain optional for small objects after clean validation; prior raw crops had costly unmatched outputs.

## 5. Evaluation and stop rules

- On the **same corrected split**, report macro mAP50–95 and mAP50 plus per-class AP, precision, recall, support, false-class confusions, missed tiny objects, and speed on the target Mac. Confusion-matrix review should explicitly include wrapper-to-bottle errors.
- Freeze validation images and annotations across candidate runs. Report improved **paired** metrics rather than comparing to a different V1 validation label set.
- Choose the final model based on V2 development data, then perform a single honest independent NHS test. If an image/class has incomplete annotations, do not call its unmatched predictions false positives.
- Preserve all V1 runs, raw datasets and label histories. A new larger model, class weight, dataset repair and new test cannot be bundled as a single causal experiment.
- No training has happened in this redesign. No measured accuracy gain is claimed.

## 6. Research sources and exactly how they apply

- Yang et al. (2020), [Object Detection as a Positive-Unlabeled Problem](https://arxiv.org/abs/2002.04672). Missing boxes can wrongly turn objects into training negatives.
- Kaul et al. (CVPR 2022), [Label, Verify, Correct](https://openaccess.thecvf.com/content/CVPR2022/html/Kaul_Label_Verify_Correct_A_Simple_Few_Shot_Object_Detection_Method_CVPR_2022_paper.html). Verify proposed pseudolabels rather than blindly training on predictions.
- Koh et al. (ICML 2021), [WILDS](https://proceedings.mlr.press/v139/koh21a.html). Domain shift can degrade deployment, motivating cafeteria-specific photos.
- Tan et al. (CVPR 2020), [Equalization Loss](https://openaccess.thecvf.com/content_CVPR_2020/html/Tan_Equalization_Loss_for_Long-Tailed_Object_Recognition_CVPR_2020_paper.html). Rare-category imbalance motivates testing supported class weighting.
- Lin et al. (ICCV 2017), [Focal Loss](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html). Foreground/background imbalance matters, but implementing a new loss is not justified as a first experiment.
- Akyon et al. (ICIP 2022), [SAHI](https://arxiv.org/abs/2202.06934). Slicing can improve small-object AP on other benchmarks, but not automatically ours.
- Ghiasi et al. (CVPR 2021), [Simple Copy-Paste](https://openaccess.thecvf.com/content/CVPR2021/html/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.html). Instance-mask-based augmentation can help rare categories; requires appropriate masks/context.
- Gupta et al. (CVPR 2019), [LVIS](https://openaccess.thecvf.com/content_CVPR_2019/papers/Gupta_LVIS_A_Dataset_for_Large_Vocabulary_Instance_Segmentation_CVPR_2019_paper.pdf). Federated evaluation uses known category coverage; imported LVIS photos are not automatically exhaustively labeled for our 11 classes.
- Wu et al. (CVPR 2022), [Entropy-Based Active Learning](https://openaccess.thecvf.com/content/CVPR2022/html/Wu_Entropy-Based_Active_Learning_for_Object_Detection_With_Progressive_Diversity_Constraint_CVPR_2022_paper.html). Select uncertain and diverse images for manual review.
- Liu et al. (ECCV 2024), [Grounding DINO](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6319_ECCV_2024_paper.php). Candidate novel-object annotation assistance, not verified truth.
- [Cleanlab object-detection error finding](https://docs.cleanlab.ai/stable/tutorials/object_detection.html). Rank annotation issues with out-of-sample predictions.
- [Ultralytics YOLO11 benchmark](https://docs.ultralytics.com/models/yolo11), [official fine-tuning guide](https://docs.ultralytics.com/guides/finetuning-guide), [training configuration](https://docs.ultralytics.com/usage/cfg). Pretrained model, freezing, class-weighting, training and evaluation controls; COCO benchmarks do not guarantee cafeteria results.
