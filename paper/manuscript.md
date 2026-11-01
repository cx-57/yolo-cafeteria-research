---
title: 'Cafeteria Waste Detection with Specialized YOLO11 Models and Selective Prediction'
author: |
  Charles Xie  
  *Natick High School*  
  Natick, Massachusetts, USA
date: ''
documentclass: article
papersize: letter
fontsize: 10pt
geometry: ['margin=0.82in']
colorlinks: false
toc: false
header-includes:
  - \usepackage{microtype}
  - \usepackage{booktabs}
  - |
    \makeatletter
    \renewcommand{\maketitle}{%
      \begin{center}
        {\LARGE\bfseries\@title\par}
        \vspace{0.7em}
        {\normalsize\@author\par}
      \end{center}
      \vspace{1em}
    }
    \makeatother
  - \setlength{\parindent}{1em}
  - \setlength{\parskip}{0pt}
---


**Abstract.** Sorting cafeteria waste requires more than recognizing discarded objects: a carton, wrapper, or food container may still require information about its material, contents, or local collection rules. We evaluate a vision-assisted prototype consisting of four specialized YOLO11 detectors for seven cafeteria-item classes and a rule-based policy that either suggests a disposal stream or requests manual review. On 248 previously examined development photographs containing 924 AI-assisted object annotations, the frozen system achieved 87.78% mAP50 and 67.24% mAP50–95. At a confidence threshold of 0.50, object-detection precision was 93.42%, with 78.35% recall. Class-specific thresholds selected using 186 validation photographs raised precision to 97.03% on validation and 97.22% on the historically exposed 248-image cohort, while reducing recall to 71.43% and 68.18%, respectively. The historical-cohort F1 also declined from 85.23% to 80.15%. Only six of those 248 images came from the participating school, and a provenance audit found no confirmed independent test cohort or human-verified disposal-bin labels. The results therefore characterize development-set recognition and its precision–recall trade-off, not the accuracy of actual composting or recycling decisions.

*Keywords:* cafeteria waste, object detection, YOLO11, selective prediction, precision–recall trade-off, evaluation provenance.

# Introduction

At the end of a school lunch period, a single cafeteria tray may contain food scraps, disposable utensils, a drink carton, and plastic packaging. Although these items are visually distinguishable, they do not necessarily belong in the same disposal stream. An item's correct destination depends not only on what it looks like, but also on its material, contamination, remaining contents, and local collection requirements. The U.S. Environmental Protection Agency advises residents to verify local recycling rules and to empty and clean recyclable containers [1]. A camera may therefore recognize an object correctly without determining how it should be discarded.

Computer vision offers a way to identify several discarded items within a single image, even when they occur together rather than against an isolated background. Object detection is especially relevant because it produces separate locations and classes for the visible items. However, the predictions of a detection model are not equivalent to verified recycling or composting decisions. A useful assistance system must distinguish between confident object recognition, uncertain predictions that warrant human review, and objects that the model fails to detect.

This study investigates that distinction through a prototype containing four specialized YOLO11 detectors and a rule-based disposal-suggestion layer. The detectors cover seven cafeteria-item classes, while the decision layer recommends composting or recycling for selected recognized categories and otherwise defers to manual review. We evaluate the frozen detectors on the available development images, examine whether class-specific confidence thresholds can reduce false-positive detections without retraining, and audit the provenance and limitations of the resulting measurements. The objective is to characterize what the existing recognition system can support, rather than to claim that its object predictions have already been validated as correct disposal decisions.

## Related Work

Early research on automated waste recognition primarily approached the problem as image classification. Thung and Yang developed TrashNet to classify photographs of individual objects into six material-based categories, including glass, metal, paper, plastic, cardboard, and trash [2]. Their work demonstrated the potential of visual waste classification, but the images feature isolated objects against relatively simple backgrounds. Single, Iranmanesh, and Raad introduced RealWaste, which contains photographs collected at a landfill and supports the study of waste classification under less controlled conditions [3]. Together, these studies show that the appearance and collection context of waste matter when evaluating whether an image classifier can generalize beyond its source dataset.

Other work has shifted from whole-image classification to locating individual objects in more complex scenes. Proença and Simões introduced TACO, a dataset with instance annotations for litter photographed in its surrounding environment, and evaluated instance segmentation with Mask R-CNN [4]. Bashkirova and colleagues further developed ZeroWaste, a dataset of highly cluttered recycling-conveyor scenes containing deformable and overlapping materials, where conventional detection and segmentation methods face substantial challenges [5]. These studies motivate object-level localization for mixed waste scenes. Nevertheless, material categories, litter-instance masks, and cafeteria disposal eligibility are different targets: even an accurately localized object may require information that is not visible in a photograph before it can be assigned to a local bin.

The object-detection approach used here follows the YOLO family, initially introduced by Redmon and colleagues as a unified framework for predicting object locations and categories from an image [6]. Standardized evaluation practices for object recognition were developed through benchmarks such as Microsoft COCO, which uses localization-sensitive average precision to assess detection quality [7]. The present system uses YOLO11 as implemented by Ultralytics [8]. These tools measure how well objects are recognized and localized, but average precision over a range of detection scores does not describe the precision or coverage of a specific operating threshold. For a decision-assistance prototype, that fixed-threshold behavior also matters.

Research on selective classification offers a framework for systems that withhold uncertain predictions. Geifman and El-Yaniv investigated rejection based on classifier confidence [9] and subsequently developed SelectiveNet, which learns a rejection option jointly with the underlying prediction task [10]. The current study does not implement or claim to outperform either approach; instead, it applies the simpler idea of confidence-based selection to already trained detectors, comparing a shared cutoff against class-specific cutoffs. This distinction is important because higher precision among accepted detections can come at the expense of missed objects and lower coverage.

Finally, evaluation design determines what those measurements can establish. The WILDS benchmark documents how performance can deteriorate when systems are tested across different sources and environments [11]. Kapoor and Narayanan similarly show that data leakage and evaluation design flaws can produce overly optimistic scientific conclusions [12]. These concerns are particularly relevant when waste images come from multiple public sources, model histories are not fully reconstructed, and nominal test images have been examined during development. Accordingly, this study separates development-set recognition results from the independent field evaluation still required to establish reliable cafeteria sorting performance.

# Data and Annotation Provenance

The working dataset consists of 2,372 training-image views, 186 validation photographs, and 248 historical test photographs. The training partition contains 2,052 original photograph entries and 320 additional augmented views. Twenty photographs from the participating school contribute 340 training views when their augmentations are included. The augmented entries expose the model to modified versions of existing images; they do not represent newly captured scenes. These counts describe the maintained working manifest, not a claim that all four inherited checkpoints were trained from scratch using exactly this partition.

The dataset combines publicly sourced waste photographs with images collected at the participating school. The 248-image historical test partition includes 73 images from `school_lunch`, 66 from `tetra_pak`, 48 from `trash_detection`, 31 from `plate`, 20 from `utensil`, four from `macworx`, and six from the school's own collection. Thus, just 6 of the 248 images (2.42%) came from the participating school; `school_lunch` is the name of a separate public dataset. The test distribution consequently does not represent a prospective sample of the school's cafeteria waste.

Object annotations were evaluated using the repaired, frozen validation and historical-test label manifests, rather than superseded label files from earlier dataset versions. The historical test set contains 924 annotated objects. The validation audit identified corrections that are absent from 109 older on-disk label files, including ten files with bottle-label differences. The frozen embedded annotations, whose identities are recorded in the experiment configuration, determine the reported results. This paper did not alter those annotations when revising its analysis.

The annotations were produced or reviewed with AI assistance, and the historical records do not establish independent human verification of every object. For example, the school review checklist contains 123 records without completed `human_reviewed` fields. Some dense food scenes also leave uncertainty about whether adjacent servings should receive separate boxes. These limits matter particularly for the hinged-container class, which has only two annotated validation objects and four historical-test objects.

**Table 1.** Annotated objects by class and split. Training counts include augmentation views and are not counts of independent photographed objects; validation and historical-test counts use the frozen repaired annotations.

| Class | Training-view boxes | Validation boxes | Test boxes |
|---|---:|---:|---:|
| Food | 2,501 | 343 | 433 |
| Plate | 722 | 110 | 123 |
| Utensil | 457 | 101 | 98 |
| Hinged container | 185 | 2 | 4 |
| Drink carton | 972 | 94 | 134 |
| Wrapper | 316 | 85 | 75 |
| Bottle | 293 | 42 | 57 |

# Model Architecture and Inference

The system consists of four separately adapted YOLO11 specialist models rather than one newly trained seven-class network. To keep this analysis reproducible, the checkpoint hashes, class mappings, inference settings, decision rules, and benchmark files were recorded in [research-freeze.json](../configs/research-freeze.json) at commit `defcc6156a4425daf1ae530757634cc541f3c267`. Protected file hashes were checked before and after the analysis. The freeze identifies the configuration evaluated in this paper, although it is not a complete inventory of every historical experiment.

**Table 2.** Frozen specialist checkpoints, parameter counts, class ownership, and per-model inference resolution.

| Specialist | YOLO11 variant | Parameters | Input size | Owned classes |
|---|---|---:|---:|---|
| food-container-carton | yolo11n.yaml | 2,590,425 | 640 | food, hinged_container, drink_carton |
| plate-wrapper | yolo11n.yaml | 2,605,126 | 640 | plate, wrapper |
| utensil | yolo11x.yaml | 56,877,241 | 800 | utensil |
| bottle | yolo11s.yaml | 9,428,179 | 640 | bottle |

Inference uses Ultralytics 8.4.175, PyTorch 2.14.1, Python 3.12, batch 1,
single-pass inference, NMS IoU 0.70, `max_det=300` per specialist, no test-time
augmentation and `half=False`. The last argument is deprecated by the installed
library but was retained to reproduce the frozen protocol. Box coordinates are
normalized XYXY. Native per-specialist NMS is followed by concatenation of the
owned classes; no additional cross-specialist deduplication is applied.

The experiments were run in a recorded environment using an Apple M1 MacBook Air with 8 GiB of memory and MPS acceleration. These specifications document the computational setting; inference time, energy use, and throughput were not benchmarked. The four checkpoints also have different training and selection histories, so the system should not be described as the product of one controlled training run.

The prediction receipt saves detections down to confidence 0.001 for AP analysis.
Display/sorting emits scores at least 0.25. Scores below 0.50 require review when
displayed; higher scores use the unchanged policy:

| Recognized class | Suggested decision at confidence >=0.50 |
|---|---|
| Food | Compost |
| Bottle, drink carton | Recycling, without checking emptiness |
| Plate, utensil, wrapper, hinged container | Review |

The policy is owner-selected, not evidence that filled containers or every detected
food/package qualifies for the school's real collection stream. Independent bin
labels must follow actual acceptance evidence rather than the detector mapping.




![System diagram: four YOLO11 specialists detect seven cafeteria classes before rule-based suggestions or review.](paper/fig-pipeline.png){width=95%}


# Evaluation Methods

For each predicted box and annotated object box, intersection-over-union (IoU) is the area of their overlap divided by the area covered by either box. The primary localization metrics are mAP50, which averages average precision (AP) across the seven classes at IoU 0.50, and mAP50–95, which additionally averages over IoU thresholds from 0.50 to 0.95 in increments of 0.05 [7]. AP is computed over confidence-ranked predictions using the frozen Ultralytics evaluation implementation. It is distinct from precision measured after selecting one confidence threshold.

At a fixed threshold, predicted objects are matched to annotations one-to-one, in descending order of confidence, when their class labels agree and IoU is at least 0.50. An unmatched prediction is a false positive, while an unmatched annotation is a false negative. We aggregate true positives (TP), false positives (FP), and false negatives (FN) across all seven classes to report micro-averaged precision, recall, and F1:

$$
P=\frac{TP}{TP+FP},\qquad
R=\frac{TP}{TP+FN},\qquad
F_1=\frac{2PR}{P+R}.
$$

The suggestion policy is assessed separately because object detection is not itself a disposal-bin decision. First, we sum class-consistent matches for the three classes eligible for an automatic suggestion: food, bottle, and drink carton. We also report a more conservative geometry-first comparison, in which all displayed detections compete for each annotated object regardless of predicted class; the predicted class is checked only after the box is assigned. This reveals cases where overlapping food and packaging predictions would otherwise each appear to be correct. Neither matching scheme establishes the correct disposal stream of an item.

The historical 248-image partition had been examined during earlier development and model-selection decisions. It is therefore described as an exposed development cohort, not a sealed independent test. Thresholds were selected using the 186-image validation cohort and then evaluated unchanged on the exposed historical cohort. In particular, selecting a stricter confidence threshold cannot change the frozen models' mAP, which is computed from the broader ranked detection outputs saved down to confidence 0.001. Test-set reuse and incomplete separation between development and evaluation can lead to optimistic performance estimates [12].

# Confidence Threshold Selection

We first examined whether a single confidence threshold could reduce false detections across all seven classes. Thresholds from 0.50 to 0.99, in increments of 0.01, and an additional threshold of 0.995 were tested on the 186-image validation set. No global threshold met all three requirements: at least 97% micro precision, at least 60% recall, and at least 100 accepted predictions. The best global threshold satisfying the recall and prediction-count requirements was 0.79, yielding 96.04% precision and 68.73% recall.

We therefore considered a separate threshold for each class. For every class, we enumerated the true and false detections accepted at candidate thresholds on validation. A dynamic program combined these class-level possibilities and selected the policy retaining the most correctly detected objects while meeting the same precision, recall, and prediction-count requirements:

$$
\max_{\boldsymbol{\tau}}\, TP(\boldsymbol{\tau})
\quad\text{subject to}\quad
P(\boldsymbol{\tau})\geq 0.97,\quad
R(\boldsymbol{\tau})\geq 0.60,\quad
N_{\mathrm{pred}}\geq100.
$$

**Table 3.** Confidence thresholds selected exclusively from development validation.

| Food | Plate | Utensil | Hinged container | Drink carton | Wrapper | Bottle |
|---:|---:|---:|---:|---:|---:|---:|
| **0.79** | **0.83** | **0.50** | **0.50** | **0.84** | **0.83** | **0.91** |

The selected thresholds were applied unchanged to the 248-image historical cohort, without another round of tuning. They were not activated in the default disposal-suggestion pipeline. Since the historical photographs had already been used during earlier development, this comparison does not provide an independent estimate of performance on new school images.


# Results

**Table 4.** Per-class fixed-threshold object-recognition metrics on the exposed
248-photo cohort (confidence ≥ 0.50, IoU ≥ 0.50); AP uses the full frozen score range.
These are descriptive development results, not independent bin classification.

| Class | TP / FP / FN | Precision | Recall | F1 | AP50 | AP50–95 |
|---|---:|---:|---:|---:|---:|---:|
| food | 327 / 10 / 106 | 97.03% | 75.52% | 84.94% | 83.91% | 55.09% |
| plate | 103 / 15 / 20 | 87.29% | 83.74% | 85.48% | 87.04% | 71.60% |
| utensil | 70 / 7 / 28 | 90.91% | 71.43% | 80.00% | 86.06% | 61.26% |
| hinged_container | 4 / 1 / 0 | 80.00% | 100.00% | 88.89% | 99.50% | 91.17% |
| drink_carton | 129 / 1 / 5 | 99.23% | 96.27% | 97.73% | 97.55% | 74.28% |
| wrapper | 58 / 7 / 17 | 89.23% | 77.33% | 82.86% | 85.13% | 57.64% |
| bottle | 33 / 10 / 24 | 76.74% | 57.89% | 66.00% | 75.27% | 59.62% |

The historical development cohort yielded 87.78% mAP50 and 67.24% mAP50–95, compared with 86.36% and 65.25%, respectively, on validation. These are macro-averaged detection scores, so each class contributes to mAP regardless of its sample size. The hinged-container AP50 of 99.50%, for example, reflects only four annotated objects in the historical cohort and is not reliable evidence of generalization for that class. At a fixed 0.50 confidence cutoff, overall historical-cohort micro precision, recall, and F1 were 93.42%, 78.35%, and 85.23%. The underlying per-class validation results appear in `results/sorting-verification.json` and [the evaluation report](../docs/sorting-evaluation.md).

**Table 5.** Automatic-suggestion object proxies, explicitly separate from bin accuracy.
Recall uses the 624 food/bottle/carton object labels.

| Matching convention | Correct object matches | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Class-specific | 489 of 510 automatic suggestions | 95.88% | 78.37% | 86.24% |
| Joint geometry-first | 483 of 510 automatic suggestions | 94.71% | 77.40% | 85.19% |

The six joint-matching conflicts are five food boxes and one bottle box assigned
to overlapping wrapper labels. They are object/matching conflicts, not independently
proven wrong-bin outcomes. The original 248-image replay was previously verified,
and a later fresh inference exactly reproduced its predictions. This finalization
reuses those receipts; it does not rerun YOLO or train another candidate.

**Table 6.** Default sorting-policy outputs at the 0.25 display-confidence floor on the exposed historical development cohort; results are not independently verified bin assignments.

| Quantity | Count / denominator | Value |
|---|---:|---:|
| Compost suggestions | 337 / 881 emitted detections | 38.25% |
| Recycling suggestions | 173 / 881 | 19.64% |
| Automatic suggestion rate | 510 / 881 | 57.89% |
| Manual-review rate among emitted detections | 371 / 881 | 42.11% |
| Geometry-matched automatic coverage of labeled objects | 489 / 924 | 52.92% |
| Exact-class correct automatic matches / labeled objects | 483 / 924 | 52.27% |
| Geometry-matched review / labeled objects | 286 / 924 | 30.95% |
| Missed objects | 149 / 924 | 16.13% |
| Unmatched automatic predictions | 21 / 510 automatic suggestions | 4.12% |
| Unmatched review predictions | 85 / 371 review predictions | 22.91% |

The default display policy emitted 881 detections at confidence 0.25 or above. Of these, 510 received a compost or recycling suggestion and 371 were marked for review, including predictions below the 0.50 automatic-suggestion cutoff and classes designated as review-only. The 42.11% review fraction applies only to emitted detections; it excludes real objects the detectors missed altogether. Likewise, the 52.27% exact-class automatic-match fraction is a detection proxy, not the proportion of waste placed in the correct bin. Conventional object-independent confidence intervals would be misleading because many objects occur in the same scenes, and the historical images were previously exposed during development.



## Precision–Recall Trade-off After Threshold Selection

**Table 7.** Overall seven-class micro object recognition at box IoU $\geq$ 0.50; same source images and labels within each cohort.

| Cohort / policy | TP | FP | FN | Precision | Recall | F1 |
|:--|--:|--:|--:|--:|--:|--:|
| Validation (186), uniform 0.50 | 638 | 48 | 139 | **93.00%** | 82.11% | 87.22% |
| Validation (186), classwise selected | 555 | 17 | 222 | **97.03%** | 71.43% | 82.28% |
| Historical (248), uniform 0.50 | 724 | 51 | 200 | **93.42%** | 78.35% | 85.23% |
| Historical (248), classwise fixed | 630 | 18 | 294 | **97.22%** | 68.18% | 80.15% |

![Precision--recall trade-off for default and validation-selected classwise confidence thresholds, measured on exposed development cohorts.](paper/fig-precision.png){width=92%}

On the historical cohort, the class-specific policy reduced false detections from 51 to 18 but also rejected 94 previously correct detections, decreasing true positives from 724 to 630. Accepted detections fell from 775 to 648. Precision increased by 3.80 percentage points, but recall fell by 10.17 points and F1 decreased from 85.23% to 80.15%. A similar F1 reduction occurred on validation (87.22% to 82.28%). Thus, the policy achieved its narrower objective of reducing false-positive detections; it did not improve the detector's overall F1 or mAP. The historical cohort was not used for a second round of threshold selection.


# Independence and Annotation Audit

A reliable evaluation requires photographs that have not influenced training, threshold selection, or earlier model decisions. The project's local inventory was therefore screened for duplicate files, overlapping capture groups, and recorded prior inference. The audit covered 102,806 readable image paths, corresponding to 67,422 distinct filesystem inodes and 14,379 byte-distinct images. Within the 9,468 candidate image identities, 3,842 were flagged for possible exposure, group overlap, or duplication; 5,482 had unresolved historical provenance; and 144 original HEIC files could not be visually decoded in the audit environment. No image was *confirmed* independent under the available evidence (Table 8). This does not prove that every candidate was used for model fitting.

The audit compared SHA-256 file identities and decoded-pixel hashes and screened potential near-duplicates using rotation/reflection-invariant comparisons of 64-bit difference hashes against 4,751 images from recorded training directories or prepared manifests. A Hamming-distance threshold of five was used for screening. These tests are deliberately conservative: membership in a training directory is not proof of optimizer exposure, while the absence of a perceptual match cannot rule out related crops, viewpoints, or capture sessions.

**Table 8.** Provenance status of the 9,468 candidate image identities. The categories reflect what could be established from available records, not a definitive partition of images actually used for training.

| Candidate status | Image identities |
|---|---:|
| Exposure, group-overlap, or duplicate flag | 3,842 |
| Unresolved historical provenance | 5,482 |
| Original HEIC visual decoding unavailable | 144 |
| Confirmed independent | 0 |

The school-specific history also prevents a prospective test claim. All 123 photographs in the earlier school submission and all 21 in the later submission shared recorded training or development capture groups. Four images from the October 29 submission entered the training collection, so that capture day was excluded from consideration as an independent test source. Earlier labels such as "sealed test" are superseded by subsequent prediction receipts and model-use records. Several audit leads overlap and must not be added together as unique training exposures.

Annotation provenance is a separate limitation. The metadata audit searched 2,398 JSON files and additional CSV, XML, and YAML files but found no confirmed set of independently human-adjudicated, per-object disposal-bin labels. A historical object-review flag is not a substitute for an actual material and disposal-eligibility annotation. The audit also noted 42 unreadable image aliases and five malformed or inaccessible JSON records, which were retained rather than silently repaired. Full provenance diagnostics and hashes are preserved in `results/research-readiness.json` and the external audit artifacts.

**Table 9.** Real-world sorting metrics that cannot be computed from the current development annotations.

| Metric | Status | Evidence required |
|---|---|---|
| Precision, recall, and F1 for actual disposal bins | Not measured | Independent images and adjudicated bin labels |
| Correct-bin accuracy | Not measured | Verified local disposal eligibility |
| Automatic coverage of independently labeled objects | Not measured | Complete, independent object annotations |
| Manual-review burden on new cafeteria photographs | Not measured | A sealed school cohort |

The lack of independently verified bin outcomes is not equivalent to a zero score. It means that the necessary ground truth does not yet exist in the audited records.

# Error Analysis and Exploratory Stress Test

The diagnostic decomposition uses frozen object labels and predictions, without
editing annotations or changing thresholds. A false-negative label is classified
as below threshold if a same-class box has IoU>=0.50 but score<0.50; a localization
candidate has same-class IoU>=0.10 but <0.50; otherwise it has no same-class overlap
at that cutoff and saved confidence floor. These are competing hypotheses, not
human-confirmed causes. A box can overlap other objects or depend on uncertain
serving definitions. Background-or-missing-annotation likewise does not distinguish
model hallucination from an incomplete label set.

**Table 10.** Frozen historical-cohort object-detection error diagnostics for food and bottle at the default 0.50 confidence cutoff. Error types are algorithmic hypotheses, not human-confirmed causes.

| Diagnostic category | Food | Bottle |
|---|---:|---:|
| Correct class matches at automatic threshold | 327 | 33 |
| FN: below automatic threshold | 53 | 19 |
| FN: localization candidate | 48 | 1 |
| FN: no same-class overlap | 5 | 4 |
| FP: localization candidate | 2 | 0 |
| FP: duplicate/competing detection | 1 | 0 |
| FP: background or missing annotation | 7 | 9 |
| FP: other-class overlap/annotation ambiguity | 0 | 1 |

Errors were concentrated in a small number of difficult scenes. One buffet photograph accounted for 52 of the 106 missed food objects (49.06%), while a single outdoor-trash photograph accounted for 10 of the 24 missed bottles (41.67%). Multiple errors in one scene should not be interpreted as independent failures across many photographs. Both scenes remained in the reported results.

AI visual inspection covered 12 deterministically selected error cases from 10
unique photographs; it is not a random audit or a human-label correction. Examples
suggest a reusable-looking sports bottle outside the disposable-bottle scope,
a possible missing bottle annotation, small cylindrical drink-package class
ambiguity, dense buffet scale/serving-granularity problems, and sealed food
packaging detected as food. Actual material, contents and bin acceptance cannot
be settled from these previews. No labels or sorting rules were changed.

A separate, previously completed bottle-specialist experiment tested one additional epoch of classification-head training. Validation AP50 declined from 82.47% to 77.95%, and AP50–95 fell from 62.00% to 59.18%. Because the candidate failed the predefined validation criterion, it was rejected without being activated or evaluated on the historical test cohort. The frozen system reported here was not retrained during manuscript finalization.

## Exploratory Public-Image Stress Test

After the research freeze, the unchanged four detectors were run on 44
source-screened public photographs (nine bottle-source, six carton-source,
15 utensil-source, 12 hinged-container-source and two wrapper-source images).
There were 149 displayed predictions at the 0.25 confidence floor: 12 compost
suggestions, 49 recycling suggestions and 88 manual-review outcomes. Three photos
had no displayed detections. These counts describe emitted predictions, not verified outcomes: there are no
human-verified object boxes, documented capture-day independence, or school bin-eligibility labels. Accordingly, no **human-verified detection mAP, correct-bin accuracy, or sorting precision** is available for the complete 44-image cohort. The separate 41-image approximate-label diagnostic below is not a substitute for that missing evaluation.

A preliminary image-level *source-hint* cross-check found at least one matching
model class on 9/9 bottle-source, 5/6 carton-source, 12/15 utensil-source,
4/12 hinged-container-source and 0/2 wrapper-source photos. These fractions
are **not recall**: source names are not exhaustive human object labels, and some
images contain multiple target items. Visual inspection suggests repeated
plate/hinged-container confusion on open foam food boxes and missing wrapper
detections. A foil meal tray was detected as `drink_carton` at confidence 0.617,
triggering an automatic recycling suggestion under the frozen user-selected
policy. This is a plausible wrong-object automatic action; actual local bin
eligibility cannot be concluded from the photo alone. A carton photo with
three cartons yielded four carton boxes and another carton image yielded none.
Crowded bottle images also exhibited overlapping boxes and cross-specialist labels.

A separate provisional comparison scored 41 of these 44 public-source photographs using approximate AI-generated object labels. The unchanged V1 detectors obtained 46.25% mAP50 and 21.16% mAP50–95; a developmental hinged-container V3 candidate obtained 44.78% mAP50 and 21.47% mAP50–95. These numbers are not comparable to a sealed test benchmark: the annotations were not independently human-verified, three crowded bottle photographs were excluded from the scoring subset, and the entire source set had already been inspected during development. The V3 candidate was not adopted as the frozen system. This comparison is included only to document an exploratory stress test, not to claim a measured out-of-distribution performance change.

These observations describe possible failure modes, not independently measured detection or disposal accuracy, and were not used to authorize a new training run.
Individual photo references and limitations are documented in
[the visual audit](https://github.com/cx-57/yolo-cafeteria-research/blob/archive/pre-cleanup-20261101/docs/exploratory-44-visual-audit.md). These public images may be
out-of-distribution relative to actual cafeteria disposal scenes, and original
source rights require review before reproducing them in a publication.



# Limitations and Future Evaluation

The main limitation of this study is that the reported benchmark images are development data, not a sealed holdout. As demonstrated in research on distribution shift and leakage, performance on reused or mismatched source data may not transfer to the intended deployment setting [11, 12]. Earlier inference, model selection, and related capture groups prevent an independent generalization claim. The object annotations were also AI-assisted, with limited evidence of human verification; in particular, the hinged-container class has only two validation objects and four historical-test objects. Food-serving boundaries and visually similar packaging introduce further annotation uncertainty. The existing audit reduces uncertainty about what data and model versions were used, but incomplete historical records mean that the absence of a detected duplicate cannot establish independence.

A second limitation concerns the distinction between an object class and a disposal decision. The current rule layer does not inspect whether a bottle or carton is empty, whether food is contaminated by packaging, or whether a specific material is accepted by the school's collection program. It cannot automatically assign landfill disposal, and requesting manual review is an abstention rather than a correct sorting outcome. The results therefore cannot quantify correct-bin accuracy or reductions in waste contamination. Latency, throughput, energy use, and user interaction during a real cafeteria deployment have also not been measured. Source-license and attribution records must be checked before any external image redistribution.

The next evaluation should use newly collected cafeteria photographs from capture sessions excluded from all training, threshold selection, and prior inspection. Inclusion criteria and capture-group membership should be fixed before model inference. Annotators who have not seen the model's predictions should mark each relevant object's bounding box and actual disposal eligibility using documented local collection rules and, when necessary, material and contents information unavailable from the photograph. Uncertain cases should be adjudicated or recorded as unscorable under a predefined protocol. On that sealed cohort, the unchanged system can then be evaluated for object detection, correct-bin precision and recall, prediction coverage, and manual-review burden, with uncertainty estimated across independent capture groups rather than treating multiple objects in one photograph as unrelated observations. For each disposal stream, an automatic wrong-bin recommendation should count against the predicted stream; a reviewed or missed item should not be credited as a successful automatic disposal decision. Reporting only accuracy among accepted decisions would conceal the effect of abstention, so the study should also report the fraction of *all* eligible labeled items assigned correctly without review.

# Conclusion

The four-specialist YOLO11 system achieved 87.78% mAP50 and 67.24% mAP50–95 on 248 historically exposed development photographs. Selecting class-specific thresholds on validation increased micro object-detection precision to 97.03% on that cohort and 97.22% on the exposed historical cohort, but reduced recall and F1. This result demonstrates a controllable precision–coverage trade-off, not a general improvement in recognition or a verified ability to direct cafeteria waste to the correct disposal stream. The next decisive experiment is evaluation of the frozen detector and disposal policy on new, independently annotated school photographs with ground-truth bin eligibility.


# Reproducibility and Data Availability

The retained implementation, experiment configuration, and summary results are maintained in the [GitHub repository](https://github.com/cx-57/yolo-cafeteria-research). The paper's frozen configuration, `configs/research-freeze.json`, records the four checkpoint hashes, class mappings, preprocessing settings, and active disposal rules. Existing benchmark outputs are in `results/sorting-verification.json`, with provenance and annotation limitations in `results/research-readiness.json`. The corresponding procedures are documented in `docs/sorting-evaluation.md`.

The underlying images, checkpoint binaries, and detailed local audit artifacts are stored separately from Git in `~/Desktop/yolo-cafeteria-research-assets/`. In particular, the combined dataset manifest and the repaired validation and historical-test annotations are external assets. Source URLs and historical licensing declarations are recorded in `data/datasets.csv` and `data/manifests/` within that asset collection, but licenses and redistribution rights have not been independently verified. The GitHub source alone is therefore insufficient to reproduce native inference, and no public release of the collected school photographs is implied.

The original audit implementation is retained in `scripts/research_audit.py`. With the external asset environment present, its file inventory can be reproduced by executing:

```sh
../yolo-cafeteria-research-assets/venv/bin/python -B scripts/research_audit.py --output /tmp/cafeteria-asset-audit.json
../yolo-cafeteria-research-assets/venv/bin/python -B -m unittest discover -s tests -v
```

These commands audit retained files and run software tests; they do not retrain YOLO or establish independent sorting accuracy. A future bin-label evaluation should use `scripts/evaluate_sorting.py` with a new, independently verified disposal-eligibility annotation file and the frozen prediction policy.

# References

[1] U.S. Environmental Protection Agency. *Frequent Questions on Recycling*. [Official recycling guidance](https://www.epa.gov/recycle/frequent-questions-recycling). Accessed November 2026.

[2] G. Thung and M. Yang. *Classification of Trash for Recyclability Status*. Stanford CS229 Project Report, 2016. [Report](https://cs229.stanford.edu/proj2016/report/ThungYang-ClassificationOfTrashForRecyclabilityStatus-report.pdf).

[3] S. Single, S. Iranmanesh, and R. Raad. *RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning*. Information, vol. 14, no. 12, article 633, 2023. [DOI: 10.3390/info14120633](https://doi.org/10.3390/info14120633).

[4] P. F. Proença and P. Simões. *TACO: Trash Annotations in Context for Litter Detection*. arXiv:2003.06975, 2020. [Preprint](https://arxiv.org/abs/2003.06975).

[5] D. Bashkirova et al. *ZeroWaste Dataset: Towards Deformable Object Segmentation in Cluttered Scenes*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21147–21157, 2022. [CVPR paper](https://openaccess.thecvf.com/content/CVPR2022/html/Bashkirova_ZeroWaste_Dataset_Towards_Deformable_Object_Segmentation_in_Cluttered_Scenes_CVPR_2022_paper.html).

[6] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi. *You Only Look Once: Unified, Real-Time Object Detection*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 779–788, 2016. [CVPR paper](https://openaccess.thecvf.com/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html).

[7] T.-Y. Lin et al. *Microsoft COCO: Common Objects in Context*. European Conference on Computer Vision, pp. 740–755, 2014. [DOI: 10.1007/978-3-319-10602-1_48](https://doi.org/10.1007/978-3-319-10602-1_48).

[8] G. Jocher and J. Qiu. *Ultralytics YOLO11*. Software, 2024. [Official model documentation and citation](https://docs.ultralytics.com/models/yolo11/).

[9] Y. Geifman and R. El-Yaniv. *Selective Classification for Deep Neural Networks*. Advances in Neural Information Processing Systems, 2017. [NeurIPS paper](https://papers.nips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html).

[10] Y. Geifman and R. El-Yaniv. *SelectiveNet: A Deep Neural Network with an Integrated Reject Option*. Proceedings of the 36th International Conference on Machine Learning, PMLR 97, pp. 2151–2159, 2019. [ICML paper](https://proceedings.mlr.press/v97/geifman19a.html).

[11] P. W. Koh et al. *WILDS: A Benchmark of in-the-Wild Distribution Shifts*. Proceedings of the 38th International Conference on Machine Learning, PMLR 139, pp. 5637–5664, 2021. [ICML paper](https://proceedings.mlr.press/v139/koh21a.html).

[12] S. Kapoor and A. Narayanan. *Leakage and the Reproducibility Crisis in Machine-Learning-Based Science*. Patterns, vol. 4, no. 9, article 100804, 2023. [DOI: 10.1016/j.patter.2023.100804](https://doi.org/10.1016/j.patter.2023.100804).

# Acknowledgments

School food services collaborated in providing cafeteria inventory context and facilitating school imagery. This does not imply that the school certified the model, its research evaluation, or actual disposal-stream eligibility.
