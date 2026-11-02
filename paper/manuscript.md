---
title: 'Specialized YOLO11 Detection and Selective Prediction for Cafeteria Waste'
author: |
  Charles Xie  
  *Natick High School*  
  Natick, Massachusetts, USA
date: ''
documentclass: article
papersize: letter
fontsize: 10pt
geometry: ['top=0.74in', 'bottom=0.78in', 'left=0.77in', 'right=0.77in']
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
        \vspace{0.48em}
        {\normalsize\@author\par}
      \end{center}
      \vspace{0.55em}
    }
    \makeatother
  - \setlength{\parindent}{1em}
  - \setlength{\parskip}{0pt}
  - \setlength{\tabcolsep}{4pt}
  - \renewcommand{\arraystretch}{1.07}
  - \setlength{\textfloatsep}{9pt plus 2pt minus 2pt}
  - \setlength{\intextsep}{8pt plus 2pt minus 2pt}
  - \setlength{\abovedisplayskip}{7pt plus 2pt minus 2pt}
  - \setlength{\belowdisplayskip}{7pt plus 2pt minus 2pt}
  - \setlength{\emergencystretch}{2em}
---


**Abstract.** Waste-sorting assistance requires more than identifying discarded objects: their appropriate disposal streams depend on material composition, contamination, contents, and local collection rules. We examine a prototype comprising four specialized YOLO11 detectors for seven cafeteria-item classes and a rule-based policy that recommends composting, recycling, or manual review. On an historically exposed development cohort of 248 photographs with 924 AI-assisted object annotations, the frozen detectors achieved 87.78% mAP50 and 67.24% mAP50–95. At a uniform confidence threshold of 0.50, micro precision and recall were 93.42% and 78.35%. Class-specific thresholds selected on 186 validation photographs increased precision to 97.03% on validation and 97.22% on the historical cohort, while reducing recall to 71.43% and 68.18%, respectively. Historical-cohort F1 consequently declined from 85.23% to 80.15%. Only six historical-cohort images were captured at the participating school, and a provenance audit identified neither a confirmed independent test cohort nor verified ground-truth disposal-bin labels. The study therefore establishes a reproducible development-set precision–coverage trade-off, not real-world disposal accuracy or reduced waste contamination.

*Keywords:* cafeteria waste, YOLO11, object detection, selective prediction, waste classification, evaluation provenance.

# Introduction

At the end of a school lunch period, a tray may contain leftover food, utensils, a drink carton, and several forms of packaging. Recognizing these items is relatively straightforward to formulate as a computer-vision problem; determining where each belongs is not. A carton may be recyclable only after it has been emptied, while packaging that appears compostable may not be accepted by the local collection program. The U.S. Environmental Protection Agency accordingly emphasizes both the need to consult local recycling guidance and the importance of emptying and cleaning containers [1]. Even a correct visual prediction can therefore lead to an inappropriate disposal recommendation.

Object detection provides a natural foundation for an assistance system because it identifies multiple items and their locations within the same photograph. Yet the confidence assigned to a detection expresses the model's assessment of its visual prediction; it does not certify an object's material, condition, or disposal eligibility. A useful system must distinguish recognition from decision-making and preserve a mechanism for withholding recommendations when the available evidence is insufficient. It must also account for objects that are never detected, since excluding uncertain predictions can make reported precision look stronger while leaving more items unaddressed.

This study evaluates a prototype combining four specialized YOLO11 detectors with a deterministic disposal-suggestion policy. Together, the detectors recognize seven item categories; the policy recommends composting or recycling for selected categories and directs others to manual review. We assess the unchanged checkpoints on available development images, investigate the precision–recall consequences of choosing separate confidence thresholds for each class, and audit the provenance of both images and labels. The central question is not whether the system has already solved cafeteria waste sorting, but how reliably its current visual predictions can support a cautious, auditable recommendation process.

## Related Work

Early waste-recognition systems generally treated the task as whole-image classification. Thung and Yang's TrashNet project classified isolated discarded objects into six material-oriented categories [2]. More recently, Single, Iranmanesh, and Raad introduced RealWaste, using photographs collected in a landfill environment to study recognition under less controlled conditions [3]. The contrast matters: a model trained on cleanly presented objects may face a substantially different visual problem when materials are dirty, crumpled, overlapping, or surrounded by unrelated waste.

Other datasets frame waste recognition as instance localization rather than image-level categorization. Proença and Simões introduced TACO, which provides annotations for litter in its surrounding context and supports instance-segmentation research [4]. Bashkirova and colleagues extended this emphasis on difficult scenes with ZeroWaste, a collection of cluttered recycling-conveyor images containing deformable, overlapping objects [5]. Both illustrate why mixed waste is poorly represented by a single label per photograph. Neither, however, makes local disposal eligibility equivalent to visual class: a detected plastic object can still require information about its material specification, contents, or collection rules before an appropriate bin can be established.

The detection architecture in this study belongs to the YOLO family introduced by Redmon and colleagues [6] and uses the YOLO11 implementations distributed by Ultralytics [8]. We follow the localization-sensitive evaluation conventions associated with Microsoft COCO, including average precision at multiple intersection-over-union thresholds [7]. These metrics summarize recognition across ranked detections, but they do not specify how a particular confidence cutoff behaves in practice. For an assistance system that acts only on selected predictions, fixed-threshold precision and recall provide complementary information.

The decision to withhold uncertain outputs connects this work to selective classification. Geifman and El-Yaniv examined confidence-based rejection [9] and subsequently introduced SelectiveNet, which integrates a rejection option into neural-network training [10]. Our method is deliberately narrower: it changes confidence cutoffs for existing detectors without retraining them or implementing a learned rejector. Higher precision among the retained predictions must therefore be assessed alongside the objects that become undetected or require review.

Finally, the reliability of any claimed improvement depends on evaluation design. WILDS documents the difficulty of transferring learned models across distributions and collection environments [11], while Kapoor and Narayanan describe how leakage can compromise reproducibility in machine-learning research [12]. These concerns are immediate here: images originate from several datasets, parts of the model-selection history remain incomplete, and the nominal historical test cohort was previously consulted during development. We consequently distinguish between observations supported by the available development records and generalization claims that would require a new, sealed evaluation.

# Data and Annotation Provenance

The maintained dataset contains 2,372 training views, 186 validation photographs, and 248 photographs in the historical test partition. Of the training views, 2,052 are original image entries and 320 are additional augmented views; twenty school photographs account for 340 views when their augmentations are included. Augmentation increases the variety of presentations seen during training without creating independently captured scenes. Because the specialist checkpoints were developed through different historical training and selection procedures, the current manifest should not be interpreted as the exact common training set for all four models.

The images come from public waste datasets and two school submissions. The historical cohort combines 73 school-lunch dataset images, 66 tetra-pak images, 48 trash-detection images, 31 plate images, 20 utensil images, four macworx images, and six photographs collected at the participating school. Thus, only 2.42% of the historical cohort consists of school-origin photographs. The separate dataset named school_lunch is not part of that school collection, and the cohort cannot be treated as a representative field sample from the cafeteria.

We score detections against the frozen, repaired annotation manifests rather than older label files that no longer reflect the accepted evaluation records. The historical cohort contains 924 annotated objects. A previous validation audit documented discrepancies in 109 older on-disk label files, including ten involving bottle annotations. None of these labels was changed for the present editorial revision; the recorded manifests and hashes remain the authoritative evaluation inputs.

Annotation quality also limits the interpretation of the scores. Historical annotations were produced or reviewed with AI assistance, and the available records do not establish comprehensive independent human verification. The school checklist, for example, contains 123 entries with blank human-review fields. Dense food scenes can also make the boundary between adjacent servings ambiguous. This uncertainty is particularly consequential for hinged containers, represented by just two validation annotations and four in the historical cohort.

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

Rather than training one model to recognize every cafeteria item, the prototype combines four previously adapted YOLO11 specialists. Each checkpoint is responsible for a defined subset of the seven output categories, as summarized in Table 2. This division reflects the retained development configuration, not a controlled comparison showing that specialization outperforms a unified detector. The checkpoints, class mappings, inference parameters, and disposal rules were recorded in the project's frozen configuration at commit defcc6156a4425daf1ae530757634cc541f3c267. The protected hashes were checked during the research audit so that the reported comparisons refer to unchanged model artifacts.

**Table 2.** Frozen specialist checkpoints, parameter counts, class ownership, and per-model inference resolution.

| Specialist | YOLO11 variant | Parameters | Input size | Owned classes |
|---|---|---:|---:|---|
| food-container-carton | yolo11n.yaml | 2,590,425 | 640 | food, hinged_container, drink_carton |
| plate-wrapper | yolo11n.yaml | 2,605,126 | 640 | plate, wrapper |
| utensil | yolo11x.yaml | 56,877,241 | 800 | utensil |
| bottle | yolo11s.yaml | 9,428,179 | 640 | bottle |

Inference was performed with Ultralytics 8.4.175, PyTorch 2.14.1, and Python 3.12. Each specialist processed one image at a time, without test-time augmentation, using its configured input resolution. Non-maximum suppression used an IoU threshold of 0.70 and a maximum of 300 detections per specialist. The saved protocol also retains half=False, an option deprecated in the installed library, to document the original invocation. Predicted boxes were represented in normalized XYXY coordinates. After each specialist's native suppression, outputs were concatenated by their assigned classes; no further cross-specialist deduplication was performed.

Evaluation took place on an Apple M1 MacBook Air with 8 GiB of memory and MPS support. These details identify the recorded computing environment rather than establish efficiency: latency, energy use, and throughput were not measured. Moreover, the four checkpoints have different adaptation and selection histories, so their combined performance should not be described as the outcome of a single controlled training experiment.

Detection outputs were retained down to confidence 0.001 for average-precision calculations. For the default display policy, predictions at or above 0.25 were shown; predictions below 0.50 were referred for review. At confidence 0.50 or above, the proposed disposal stream followed the class-dependent rules in the table below.

| Detected category | Default suggestion at confidence ≥ 0.50 |
|---|---|
| Food | Compost |
| Bottle or drink carton | Recycling |
| Plate, utensil, wrapper, or hinged container | Manual review |

These rules encode a provisional decision policy, not independently verified disposal eligibility. In particular, the recycling suggestion does not establish that a container is empty or accepted by the school's collection program. The system does not issue an automatic landfill recommendation.

![Processing pipeline: four YOLO11 specialists produce seven item classes, which are mapped to provisional compost, recycling, or manual-review suggestions.](paper/fig-pipeline.png){width=95%}

# Evaluation Methods

We evaluate localization and classification jointly. Intersection-over-union (IoU) measures the overlap between a predicted bounding box and its reference annotation, divided by the area of their union. The primary ranking-based measures are mAP50, averaged across the seven classes at IoU 0.50, and mAP50–95, which additionally averages over thresholds from 0.50 to 0.95 in steps of 0.05 [7]. Both are calculated from confidence-ranked detections using the retained Ultralytics evaluation procedure; neither represents precision at one operational cutoff.

For fixed-threshold evaluation, detections are considered in descending confidence order and matched one-to-one to unused annotations of the same class when IoU is at least 0.50. Matched detections count as true positives (TP), unmatched detections as false positives (FP), and unmatched annotations as false negatives (FN). Aggregating these counts over all classes gives micro-averaged precision, recall, and F1:

$$
P=\frac{TP}{TP+FP},\qquad
R=\frac{TP}{TP+FN},\qquad
F_1=\frac{2PR}{P+R}.
$$

We distinguish these detection measures from two object-level proxies for automatic disposal suggestions. The first counts class-consistent matches separately for food, bottle, and drink carton. The second assigns displayed detections to reference boxes by geometry before checking their classes, thereby exposing competition between overlapping predictions from different specialists. A prediction that matches a food box but conflicts with a wrapper annotation illustrates why these approaches can yield different totals. Neither proxy can establish whether the object actually belongs in the suggested bin.

The evaluation also distinguishes development evidence from independent generalization. The 248-image historical cohort was examined during earlier model development and selection, so it cannot be treated as a sealed holdout. Class-specific thresholds were determined from the separate 186-image validation cohort and then applied unchanged to those historical images. Since thresholding only filters an existing ranked set of predictions, it changes fixed-cutoff precision and recall but does not change the frozen detectors' mAP. Previous use of evaluation data can otherwise inflate apparent performance [12].

# Confidence Threshold Selection

The initial experiment tested whether one confidence cutoff could improve precision while preserving useful detection coverage. On the 186-image validation cohort, we evaluated global thresholds from 0.50 to 0.99 at 0.01 intervals, together with 0.995. No single threshold simultaneously achieved the three prespecified criteria of at least 97% micro precision, at least 60% recall, and at least 100 accepted predictions. Among the global cutoffs satisfying the latter two requirements, the highest precision was 96.04% at a threshold of 0.79, with 68.73% recall.

We then allowed each class to have its own confidence threshold. Candidate cutoffs for each class were summarized by their validation-set true- and false-positive counts. A dynamic program combined these classwise choices, preserving the largest attainable number of true positives for each feasible false-positive total. The final threshold vector maximized correctly accepted detections subject to the same precision, recall, and accepted-prediction constraints:

$$
\max_{\boldsymbol{\tau}}\,TP(\boldsymbol{\tau})
\quad \text{subject to}\quad
P(\boldsymbol{\tau})\geq0.97,\quad
R(\boldsymbol{\tau})\geq0.60,\quad
N_{\mathrm{pred}}\geq100.
$$

**Table 3.** Class-specific confidence thresholds selected on development validation.

| Food | Plate | Utensil | Hinged container | Drink carton | Wrapper | Bottle |
|---:|---:|---:|---:|---:|---:|---:|
| 0.79 | 0.83 | 0.50 | 0.50 | 0.84 | 0.83 | 0.91 |

The selected vector was subsequently applied without modification to the historical cohort. It was not used to replace the existing disposal-policy settings, and no additional cutoff selection was performed on those historical images. Because those images had already informed development, the resulting comparison remains descriptive rather than an independent estimate of field performance.

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

The historical cohort achieved 87.78% mAP50 and 67.24% mAP50–95; the corresponding validation scores were 86.36% and 65.25%. These macro-averages give each class equal weight, regardless of its annotation count. The hinged-container AP50 of 99.50%, for example, is based on only four historical-cohort objects and should not be interpreted as robust performance for that category. At the uniform 0.50 confidence cutoff, pooled detection precision, recall, and F1 on the historical cohort were 93.42%, 78.35%, and 85.23%, respectively. Detailed validation results are available in the frozen evaluation records.

**Table 5.** Automatic-suggestion object proxies, explicitly separate from bin accuracy.
Recall uses the 624 food/bottle/carton object labels.

| Matching convention | Correct object matches | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Class-specific | 489 of 510 automatic suggestions | 95.88% | 78.37% | 86.24% |
| Joint geometry-first | 483 of 510 automatic suggestions | 94.71% | 77.40% | 85.19% |

Class-specific matching counted 489 correct object detections among 510 automatic suggestions; geometry-first matching retained 483. Five food detections and one bottle detection were instead assigned to overlapping wrapper labels under the latter convention. These are disagreements in object assignment, not verified wrong-bin outcomes. The frozen 248-image predictions were reproduced in a separate inference replay, and the present analysis uses those saved outputs without retraining the models.

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

At the display threshold of 0.25, the default policy emitted 881 detections: 337 compost suggestions, 173 recycling suggestions, and 371 manual-review outcomes. Automatic suggestions therefore accounted for 57.89% of emitted detections. This denominator is narrower than the set of actual objects in the photographs, since undetected items generate no suggestion or review record. Indeed, exact-class automatic matches account for 483 of the 924 annotated objects (52.27%), a coverage-sensitive detection proxy rather than correct-bin accuracy. We do not attach conventional object-independent confidence intervals to these results: predictions are clustered within photographs, annotations contain unresolved uncertainty, and the historical cohort has prior development exposure.

## Precision–Recall Trade-off After Threshold Selection

**Table 7.** Overall seven-class micro object recognition at box IoU $\geq$ 0.50; same source images and labels within each cohort.

| Cohort / policy | TP | FP | FN | Precision | Recall | F1 |
|:--|--:|--:|--:|--:|--:|--:|
| Validation (186), uniform 0.50 | 638 | 48 | 139 | **93.00%** | 82.11% | 87.22% |
| Validation (186), classwise selected | 555 | 17 | 222 | **97.03%** | 71.43% | 82.28% |
| Historical (248), uniform 0.50 | 724 | 51 | 200 | **93.42%** | 78.35% | 85.23% |
| Historical (248), classwise fixed | 630 | 18 | 294 | **97.22%** | 68.18% | 80.15% |

![Precision--recall trade-off for default and validation-selected classwise confidence thresholds, measured on exposed development cohorts.](paper/fig-precision.png){width=92%}

The classwise policy reduced historical-cohort false positives from 51 to 18, but also removed 94 true positives, from 724 to 630. The number of accepted predictions declined from 775 to 648. Consequently, precision increased by 3.80 percentage points while recall decreased by 10.17 points; F1 fell from 85.23% to 80.15%. Validation F1 likewise fell from 87.22% to 82.28%. The result supports a specific trade-off—fewer false-positive detections among accepted outputs—rather than an overall improvement in recognition quality. Neither the model weights nor the ranking-based mAP changed.


# Independence and Annotation Audit

The principal threat to independent evaluation is the possibility that photographs, closely related capture sessions, or earlier prediction results influenced model selection. We therefore examined the retained local inventory rather than inferring independence from folder names. The audit identified 102,806 readable image paths, 67,422 distinct filesystem inodes, and 14,379 byte-distinct images. Among 9,468 candidate identities, 3,842 were flagged for possible exposure, capture-group overlap, or duplication; 5,482 had unresolved historical provenance; and 144 original HEIC files could not be visually decoded in the audit environment. None could be positively certified as independent under the available records (Table 8), although this does not establish that every candidate was actually used to train a checkpoint.

To identify potentially repeated imagery, the audit compared file-level SHA-256 and decoded-pixel hashes, then screened near-duplicates using rotation- and reflection-aware comparisons of 64-bit difference hashes at a Hamming-distance threshold of five. The screening pool contained 4,751 images associated with training directories or prepared manifests. These checks provide conservative evidence of possible overlap, not exhaustive proof of model exposure: directory membership does not guarantee that a particular optimizer consumed an image, while nonmatching hashes cannot exclude related crops, viewpoints, or photographs taken in the same session.

**Table 8.** Provenance status of the 9,468 candidate image identities. The categories reflect what could be established from available records, not a definitive partition of images actually used for training.

| Candidate status | Image identities |
|---|---:|
| Exposure, group-overlap, or duplicate flag | 3,842 |
| Unresolved historical provenance | 5,482 |
| Original HEIC visual decoding unavailable | 144 |
| Confirmed independent | 0 |

Both school submissions have development-related provenance. The earlier submission contained 123 photographs and the later one 21; each set shared recorded training or development capture groups. Four images from the October 29 submission entered the training collection, excluding that capture day from independent evaluation. Historical descriptions such as “sealed test” must therefore be interpreted alongside later inference receipts and model-use records. Overlap counts describe different audit indicators and cannot be summed as if they identified disjoint training exposures.

The audit separately examined annotation provenance, screening 2,398 JSON files together with additional CSV, XML, and YAML records. It did not identify a confirmed set of independently human-adjudicated, object-level disposal-bin labels. An annotation marked “reviewed” in an earlier workflow is not, by itself, evidence that a human established the object's material or appropriate bin. Forty-two unreadable image aliases and five malformed or inaccessible JSON records were also documented rather than silently repaired. The complete provenance summary and file identities are retained in the research-readiness report and external audit artifacts.

**Table 9.** Real-world sorting metrics that cannot be computed from the current development annotations.

| Metric | Status | Evidence required |
|---|---|---|
| Precision, recall, and F1 for actual disposal bins | Not measured | Independent images and adjudicated bin labels |
| Correct-bin accuracy | Not measured | Verified local disposal eligibility |
| Automatic coverage of independently labeled objects | Not measured | Complete, independent object annotations |
| Manual-review burden on new cafeteria photographs | Not measured | A sealed school cohort |

The absence of these metrics should not be interpreted as zero performance. They are undefined because the audited collection does not contain the independent disposal-label evidence needed to compute them.

# Error Analysis and Exploratory Stress Test

We examined the frozen predictions for food and bottle, two classes that contribute to automatic disposal suggestions. A missed object was assigned a diagnostic category according to the strongest same-class overlap in the saved outputs: a sufficiently localized detection below confidence 0.50, a candidate overlapping at IoU 0.10–0.49, or no qualifying same-class overlap. False positives were similarly grouped by localization, duplication, competing class, or lack of a corresponding annotation. These categories are algorithmic descriptions rather than verified causal explanations. In particular, an apparent background false positive could reflect an incomplete reference annotation.

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

Errors were concentrated in a few demanding photographs. One buffet scene accounted for 52 of 106 missed food objects (49.06%), while a single outdoor-trash scene accounted for 10 of 24 missed bottles (41.67%). The totals therefore reflect substantial within-image clustering rather than independent failures across dozens of different scenes. Both photographs were retained in the evaluation.

A targeted AI-assisted visual inspection considered 12 error cases from ten unique photographs. The examples included a sports-style bottle outside the intended disposable-container scope, a possible unannotated bottle, ambiguous cylindrical packaging, crowded food portions, and sealed food packaging recognized as food. These observations help identify plausible annotation and class-boundary problems, but cannot determine actual material, contents, or bin acceptance from the available previews. No annotations or disposal rules were modified.

A previously completed bottle-specialist trial also provided a negative experimental result. One additional epoch of classification-head training decreased validation AP50 from 82.47% to 77.95% and AP50–95 from 62.00% to 59.18%. The candidate failed its predefined validation criterion and was neither activated nor scored on the historical cohort. All main-paper results continue to use the frozen bottle checkpoint.

## Exploratory Public-Image Stress Test

The unchanged detectors were also applied to 44 source-screened public photographs: nine from bottle sources, six from carton sources, fifteen from utensil sources, twelve from hinged-container sources, and two from wrapper sources. At confidence 0.25, the system emitted 149 predictions—twelve compost suggestions, 49 recycling suggestions, and 88 review outcomes—while three photographs produced no displayed detection. These are output counts, not measurements against independently verified object or disposal labels.

A preliminary source-category check found at least one detection of the expected source class in 9/9 bottle photographs, 5/6 carton photographs, 12/15 utensil photographs, 4/12 hinged-container photographs, and 0/2 wrapper photographs. These proportions are not class recall: the source categories do not provide exhaustive instance-level ground truth, and several photographs contain multiple objects. Visual examination suggested repeated confusion between open foam containers and plates, missed wrappers, and overlapping predictions in crowded bottle scenes. In one example, a foil meal tray was classified as a drink carton with confidence 0.617 and consequently received a recycling suggestion; the photograph alone does not establish its correct local disposal stream.

A separate, provisional comparison used approximate AI-generated boxes for 41 of the 44 photographs, excluding three crowded bottle scenes. The frozen V1 model ensemble achieved 46.25% mAP50 and 21.16% mAP50–95, while a developmental hinged-container V3 candidate achieved 44.78% and 21.47%, respectively. Because the annotations were not independently human verified and the photographs had already been inspected during development, neither comparison constitutes a valid independent benchmark. The V3 candidate was not incorporated into the frozen system.

Taken together, these observations illustrate plausible failure modes and the limits of the available annotation process, not demonstrated field accuracy or a validated improvement under distribution shift. Case-level notes are preserved in [the archived visual audit](https://github.com/cx-57/yolo-cafeteria-research/blob/archive/pre-cleanup-20261101/docs/exploratory-44-visual-audit.md). The source images require rights verification before reproduction in a publication.

# Limitations and Future Evaluation

The first limitation is evaluation independence. Although the model configuration and reported outputs are frozen, the historical cohort was consulted during earlier development, and the full lineage of every candidate photograph cannot be reconstructed. AI-assisted object annotations introduce further uncertainty, particularly in scenes where serving boundaries or similar forms of packaging are ambiguous. These factors weaken claims of generalization even when a benchmark's numerical precision appears high. The broader literature on distribution shift and leakage underscores why reused or mismatched evaluation data may overstate performance in a new setting [11, 12].

A second limitation is the gap between object recognition and sorting correctness. The policy does not establish whether bottles and cartons are empty, whether packaging contaminates food scraps, or whether the local waste contractor accepts a particular material. Nor does the system automatically recommend landfill disposal. Manual review is an abstention, not a successful sort, while undetected objects are omitted from the displayed review rate. No evidence was collected about contamination reduction, user compliance, inference latency, or energy consumption in a live school deployment. Public-dataset licensing and attribution must also be verified before any image redistribution.

An independent field study would require photographs collected in previously unused school sessions, with inclusion criteria and capture-group boundaries fixed before inference. Human annotators blinded to model predictions should identify each eligible object and document its actual disposal destination according to school-specific collection rules and, where necessary, material or contents evidence beyond the photograph. Disagreements and uncertain items should undergo recorded adjudication. The frozen checkpoints, thresholds, and decision rules could then be evaluated without further tuning, with class-specific detection metrics, correct-bin precision and recall, manual-review workload, and automatic coverage reported together. In addition to accuracy among accepted recommendations, the study should report the proportion of all eligible objects correctly assigned without review; otherwise, abstention could make a weak system appear artificially accurate. Uncertainty should be estimated across independent capture groups rather than treating objects from one image as independent observations.

# Conclusion

The four-specialist YOLO11 system attained 87.78% mAP50 and 67.24% mAP50–95 on the available 248-image historical development cohort. Selecting confidence thresholds separately for each class increased measured micro precision to 97.03% on validation and 97.22% on the exposed historical cohort, but at the expense of lower recall and F1. The contribution is therefore a transparent account of selective detection and its coverage costs, supported by a frozen evaluation and an explicit provenance audit. It is not evidence that the system reliably directs waste to the correct bin. Establishing that capability will require an independent school-based study with human-verified object and disposal labels.

# Reproducibility and Data Availability

The [project repository](https://github.com/cx-57/yolo-cafeteria-research) retains the inference code, class mappings, disposal rules, evaluation scripts, and summary results. The recorded configuration in configs/research-freeze.json identifies the four checkpoint hashes and fixed inference settings; results/sorting-verification.json contains the retained detection results, and results/research-readiness.json summarizes the provenance audit. Evaluation conventions and label requirements are described in docs/sorting-evaluation.md.

The complete image data, checkpoint binaries, and detailed local audit artifacts are held outside Git in the project's accompanying asset directory. The working data manifest and frozen repaired annotation files are likewise external. As a result, the public source code alone does not reproduce native inference. Historical source and license declarations are retained with the assets, but attribution and redistribution permissions have not been independently audited; the school photographs are not presented as a released dataset.

With the external environment available, the existing inventory and automated software tests can be executed using:

\`\`\`sh
../yolo-cafeteria-research-assets/venv/bin/python -B scripts/research_audit.py --output /tmp/cafeteria-asset-audit.json
../yolo-cafeteria-research-assets/venv/bin/python -B -m unittest discover -s tests -v
\`\`\`

These commands neither retrain the detectors nor generate independent field-performance metrics. The retained scripts also support future evaluation of separately collected, human-verified disposal labels without altering the frozen model configuration.

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

The participating school's food services staff provided cafeteria inventory context and assisted with image collection. Their involvement does not imply institutional verification of the annotations, measured performance, or disposal recommendations.
