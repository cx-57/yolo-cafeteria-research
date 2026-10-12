# Small objects, occluded plates, and visible-box policy — October 12, 2026

## What was actually tested (no retraining)

The user-supplied 100-epoch checkpoint (`yolo11n-corrected-baseline-100/weights/best.pt`) was evaluated locally with pinned Ultralytics 8.4.175 on **71 original GitHub public-source validation images**, using the source annotations and *not* the 619-image full validation split.

| Class | 640px mAP50 | 960px mAP50 |
|---|---:|---:|
| food | 0.995 | 0.976 |
| plate | **0.473** | **0.470** |
| utensil | 0.698 | 0.823 |
| drink_carton | 0.995 | 0.995 |
| wrapper | **0.482** | **0.444** |
| bottle | **0.571** | **0.512** |

640px test-time augmentation did **not** improve overall performance: mini-cohort mAP50 **0.680**, compared to **0.702** without TTA. Increasing input size alone is not a reliable repair. Results above are sample-specific and must not be presented as NHS real-world accuracy or compared to an independently sampled set.

### Measured size dependence

On 55 validation images selected **because they contained plates, wrappers, or bottles**, predictions were matched one-to-one by class at **confidence 0.25 and box IoU ≥ 0.50**. Results here are fixed-threshold target matches, **not AP or a substitute for the official evaluator**.

| Target class | Box occupies < 1% of image | Box occupies ≥ 16% of image |
|---|---:|---:|
| Plate | **6/26** targets matched | **12/13** targets matched |
| Bottle | **3/20** targets matched | **1/1** target matched |
| Wrapper | **0/3** targets matched | **1/1** target matched |

Some source annotations may be incomplete; small GT box size is not proof of occlusion. These counts distinguish **visible size** from the separate question of whether an object is partially concealed.

### Sliced/cropped inference experiment

On exactly the same 55 validation photos and at confidence 0.25 / IoU 0.50, a 60%-width/height 2×2 crop grid was run at 640px, crop predictions were remapped to full-image coordinates, and class-wise NMS merged them with whole-image predictions. These results are **fixed-threshold P/R**:

| Class | Whole-image matched/total | Whole-image precision | Whole + crops matched/total | Whole + crops precision |
|---|---:|---:|---:|---:|
| Plate | 30/57 | 0.566 | 34/57 | **0.337** |
| Wrapper | 6/19 | 0.750 | 7/19 | 0.778 |
| Bottle | 16/35 | 0.696 | 19/35 | **0.452** |

**Decision: do not enable naive slicing**. Increased recall is accompanied by unacceptable numbers of new false detections, particularly for plates and bottles. A separate 960px whole-image pass and TTA were also unsuccessful at improving these classes. Tuning against this small cohort would risk overfitting the validation set.

## How to annotate partly hidden plates (critical)

**Keep occluded or edge-clipped plates.** Do not remove these cases, arbitrarily relabel them as background, or shrink every plate box based on a generic rule.

Two different annotation definitions must **not** be mixed:

- **Visible-extent bounding box:** a tight axis-aligned rectangle around the pixels known to belong to the plate, including disjoint exposed portions. If food covers the plate center but the rim is visible on both sides, the box can still span both rim pieces and the intervening food. It is *not* a polygon that excludes the food.
- **Amodal/full-object bounding box:** estimates the hypothetical entire plate extent behind occluding objects. Do not switch to this convention or combine it with visible-extent labels without tracking provenance.

The existing plate imports in `scripts/lvis.py` copy the official LVIS `annotation["bbox"]` directly; they **do not synthesize a box around an imagined hidden plate**. The stored rectangles often already encompass the visible object extent. However, the source bbox–segmentation consistency has not been exhaustively verified. Do not announce that globally replacing boxes with smaller boxes is a new fix.

**Shape is a different output:** YOLO11n *detection* predicts rectangles, not a visible plate silhouette. A plate segmentation model trained on **pixel-level polygons/masks** could represent a partly covered rim more precisely. It would require actual mask annotations and controlled evaluation, not geometric inference from rectangles; more precise masks do not automatically guarantee higher detection mAP.

### Review standards

1. For each suspicious plate, visually compare the *original* image, official plate box, and any source polygon if available. Record whether the bounding rectangle is wrong, whether a separate plate is unlabeled, or whether the detection merely misses a tiny/occluded correct target.
2. Correct confirmed box or class errors only in a **separate audited experiment**; include provenance, keep all original photos, and never reuse validation examples for training.
3. If only an exposed sliver of a plate is visible, record its visible extent and whether it is distinguishable. Avoid inventing the invisible geometry.
4. Check annotated **food inside plates**: a single plate photo with omitted food labels creates conflicting supervision. The existing LVIS-food additive experiment accepted 1,355 nonduplicate food boxes on the frozen image membership, including 551 on validation examples. It has **not** been proven to improve detection.
5. Prioritize **new, naturally small cafeteria objects** and realistic partly occluded plates in *training*, with capture-group separation and exhaustive labels. Repeated views of the same photos are not independent data.
6. Reserve completely independent NHS scenes for final accuracy estimates. Paper boats have no labeled source images. Napkins, condiment packets and bowls are not evaluable on the current validation split.

## New non-training diagnostic

`scripts/inspect_predictions.py` now samples pictures across box-area bins instead of simply selecting the earliest filenames. It writes a `review_queue.csv` of missed targets / unmatched predictions and `summary.json` with per-class and size-bin counts. The script never edits training data or weights.

```bash
python3 scripts/inspect_predictions.py \
  --weights model/runs/yolo11n-corrected-baseline-100/weights/best.pt \
  --data data/training/data.yaml --split val --classes plate wrapper bottle \
  --per-class 24 --conf 0.25 --iou 0.5 --device mps \
  --output model/prediction-review-size
```

A future **short** warm-start experiment should change only one factor at a time (e.g., revised multi-class plate-scene annotations *or* a reviewed batch of partly occluded plate examples). Compare to an equal-budget baseline **on identical frozen labels and split**, measuring native AP50/AP50–95 and error rates for small plates, wrappers, and bottles. **Do not schedule another 100-epoch run until such an experiment demonstrates improvement.**

### What remains unproven

No updated weights were trained. No new plate mask dataset was produced. No 95% mAP was achieved. Every improvement hypothesis above requires prospective training and a held-out measurement.
