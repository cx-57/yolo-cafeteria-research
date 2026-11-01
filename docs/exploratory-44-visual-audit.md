# Exploratory 44-photo visual audit — November 1, 2026

## Status and scope

This report is a **qualitative visual review of the 44 frozen-model overlay images** supplied by the project owner as `exploratory-results.zip` (44 JPEG overlays, `detections.csv`, `summary.json`). The photos were sourced from public image collections and source-screened in the earlier review. No object ground truth or verified actual school bin labels were supplied. These observations **are not precision, recall, mAP, correct-bin accuracy, or an independent evaluation**. Source-category names are only *hints*, not true object annotations. This review makes no changes to frozen checkpoints, class mapping, thresholds, or source data.

## Model-generated totals

- Photos: **44**; displayed predictions (score at least 0.25): **149**.
- Images with no displayed predictions: **3**.
- Predicted classes: bottle **51**, drink_carton **15**, food **28**, hinged_container **8**, plate **25**, utensil **21**, wrapper **1**.
- Policy output: **12 compost**, **49 recycling**, **88 review**; **61/149** predictions (40.9%) issued automatic bin suggestions. Those are *outputs*, not correctly sorted items.
- No verified object labels or true bins; all actual accuracy values remain **undefined**.

### Source-hint class presence (not detection recall)

A *hit* here means that an image from the named source-hint group had **at least one** corresponding predicted class at the display threshold. This is *not* sensitivity/recall: some images contain multiple targets, some class hints can be ambiguous, and annotations are not human certified.

| Public source-hint group | Photos | Photos with matching model class |
|---|---:|---:|
| Bottle | 9 | 9 |
| Drink carton | 6 | 5 |
| Utensil | 15 | 12 |
| Hinged container candidate | 12 | 4 |
| Wrapper | 2 | 0 |

The **only wrapper detection** across all 44 photos occurred in a *drink-carton* image, on the visible hand/adjacent background rather than on a verified wrapper. The five `hinged_container` predictions in hinged-source scenes came from just four photos (one photo had two overlapping boxes). Three additional hinged predictions occurred in bottle scenes.

## Most consequential visible cases

These are **possible/visually evident errors**, not ground-truth-confirmed false positive/negative counts.

| Image identifier (prefix) | Observation from provided overlay | Research relevance |
|---|---|---|
| `commons__hinged_container_candidate_003` | Open foam clamshell with packaged food, **no detections** | Visible missed container |
| `commons__hinged_container_candidate_001`, `004`, `007`, `008`, `011`, `012`, `015` | Foam clamshells mainly yield food and/or plate predictions, with **no hinged_container class** | Repeated plate-vs-hinged confusion, especially on open food-filled boxes |
| `commons__hinged_container_candidate_016` | One open foam clamshell receives **two** hinged-container boxes (0.83/0.63) | Apparent part/whole double-counting |
| `commons__wrapper_001` | Pink snack package, **no detections** | Clear visually missed wrapper category |
| `commons__wrapper_005` | Flat blue snack wrapper has **no wrapper prediction**; instead bottle at 0.41 | Apparent cross-class false detection, but threshold routes it to review |
| `commons__utensil_009` | Packaged utensil visible, **no utensil prediction** | Utensils in packaging may be missed |
| `commons__utensil_010` | Four standalone disposable utensils, only **two utensil boxes** | Misses despite a simple background |
| `commons__utensil_006` | Drinking straws apparently marked as utensils; bottle class on a glass bottle at 0.40 | Potential class-definition confusion |
| `commons__utensil_015` | **Foil food tray** receives a `drink_carton` detection at **0.617**, triggering an automatic **recycling** suggestion | Most important observed high-confidence wrong-class automatic suggestion; real bin eligibility of the tray remains unverified |
| `trashbox__drink_carton_14` | Upright capped milk carton, **no detections** | Visible missed carton |
| `trashbox__drink_carton_02` | Three foreground cartons but **four** carton detections, including a box above the cartons at 0.667 | Likely extra/poorly localized automatic recycling suggestion |
| `trashbox__drink_carton_09` | Correctly detects milk carton, but also predicts `wrapper` 0.742 on the hand/background | Spurious wrapper detection |
| `trashbox__bottle_03` | Correct plastic bottle box plus `hinged_container` 0.60 on part of bottle | Wrong-class box; maps to review |
| `trashbox__bottle_08` | One held bottle receives two overlapping bottle boxes (0.54/0.37) | Apparent duplicate; second output becomes review |
| `trashbox__bottle_09` | Bottle shelf scene produces bottle boxes plus hinged-container and drink-carton predictions on bottles | Cross-specialist confusion in multiple-object scenes |
| `trashbox__bottle_10` | Dense pile of bottles produces 26 boxes but no complete ground truth | Exclude from any informal recognition-rate denominator; many overlapping and tiny targets |

## Interpretation and next decision

1. **Bottles:** visually plausible detection on most single-bottle scenes; crowding creates duplicate boxes and cross-specialist false labels. Avoid claiming strong bottle recall from this small public sample.
2. **Cartons:** several clean examples detect well; one clear miss (`drink_carton_14`) and an extra detection in the three-carton photo. One **false carton prediction on a foil food tray with score 0.62** exposes risk in automatic recycling.
3. **Utensils:** many obvious forks/spoons/knives are detected, but enclosed utensils and some isolated white utensils are missed. Sticks/straws may trigger this class even though the configured class refers to forks, knives and spoons.
4. **Hinged containers:** repeated substitution with `plate` (and `food`) on open food-filled foam clamshells; one image with no predictions. This class merits targeted work *if starting a separate development iteration*.
5. **Wrappers:** **0/2** source-hint wrapper photos triggered the wrapper class; the only wrapper output in the cohort looks wrong. The sample is too small to quantify failure but warrants focused qualitative investigation.

**Recommendation:** preserve the current frozen research version. This is evidence of potential failure modes, **not** proof that retraining on these 44 photos will improve performance. Do not contaminate a future independent test cohort by training on its images. A separate, explicitly labeled development iteration could gather additional wrapper and hinged-container *training* examples, add hard negatives for foil trays and bottles, and test overlap handling. Judge proposed changes on a distinct verified validation set and then new field captures, not on source-hint counts alone.

For a prototype manuscript, report these observations in a clearly labeled *exploratory public-image stress test* appendix; retain the existing exposed-development mAP figures with their limitations, and do not claim independent real-world sorting accuracy.
