---
title: 'Detecting Cafeteria Waste with Specialized YOLO11 Models'
author: |
  Charles Xie  
  *Natick High School*  
  Natick, Massachusetts, USA
date: ''
documentclass: article
classoption: [twocolumn]
papersize: letter
fontsize: 10pt
geometry: ['margin=0.72in', 'columnsep=0.24in']
colorlinks: false
toc: false
header-includes:
  - \usepackage{times}
  - \usepackage{microtype}
  - \usepackage{tabularx}
  - \usepackage{booktabs}
  - \usepackage{array}
  - \usepackage{graphicx}
  - \usepackage{caption}
  - \captionsetup{font=small,labelfont=bf}
  - \IfFileExists{balance.sty}{\usepackage{balance}}{}
  - \setlength{\parindent}{1em}
  - \setlength{\parskip}{0pt}
  - \setlength{\textfloatsep}{8pt plus 2pt minus 2pt}
  - \setlength{\floatsep}{7pt plus 2pt minus 2pt}
  - \setlength{\intextsep}{7pt plus 2pt minus 2pt}
  - \setlength{\emergencystretch}{2em}
  - |
    \makeatletter
    \def\@maketitle{%
      \newpage\null\vskip .5em
      \begin{center}
        {\LARGE\bfseries\@title\par}
        \vskip .55em
        {\normalsize\@author\par}
      \end{center}
      \vskip .45em
    }
    \renewcommand{\abstractname}{\textbf{Abstract}}
    \renewcommand\section{\@startsection{section}{1}{\z@}{-1.8ex plus -.4ex minus -.1ex}{.6ex plus .1ex}{\normalfont\large\bfseries}}
    \renewcommand\subsection{\@startsection{subsection}{2}{\z@}{-1.2ex plus -.3ex minus -.1ex}{.4ex plus .1ex}{\normalfont\normalsize\bfseries}}
    \makeatother
---


\begin{abstract}
In school cafeterias, discarded food, packaging, and utensils are often placed together, making accurate waste identification difficult. A central challenge in building an automated sorting assistant is that recognizing an object does not necessarily establish which disposal stream accepts it. This study examines four specialized YOLO11 detectors that identify seven cafeteria-item classes, together with a rule-based policy for compost, recycling, and manual-review suggestions. The system achieved 87.78\% mAP50 and 67.24\% mAP50--95 on 248 previously examined development photographs containing 924 AI-assisted annotations. At a uniform confidence threshold of 0.50, object-detection precision and recall were 93.42\% and 78.35\%. Class-specific thresholds selected on 186 validation photographs increased precision to 97.03\% on validation and 97.22\% on the historical cohort, while reducing recall to 71.43\% and 68.18\%, respectively. However, the historical images had been exposed during development, only six came from the participating school, and no independently verified disposal-bin labels were available. These results demonstrate a precision--recall trade-off in cafeteria-item detection, not verified waste-sorting accuracy.
\end{abstract}

*Keywords:* object detection, YOLO11, cafeteria waste, selective prediction, recycling.

# Introduction

School cafeterias produce a mixture of food scraps, disposable utensils, drink cartons, and packaging. Although these objects can often be identified from photographs, their appropriate disposal streams depend on additional information, including material, contamination, and local collection rules. The U.S. Environmental Protection Agency recommends checking local recycling requirements and emptying and cleaning containers before recycling [1]. Consequently, correct object recognition does not necessarily imply a correct sorting decision.

Object detection offers a way to identify several discarded items in one image, rather than assigning a single label to the entire scene. However, a detector's confidence score concerns the predicted object class, not whether the object belongs in a particular bin. A stricter confidence threshold may reduce false detections, but it may also discard correct ones. An effective assistance system must therefore consider the trade-off between precision and recall while allowing uncertain cases to be reviewed.

This study evaluates four specialized YOLO11 detectors covering seven cafeteria-item classes, together with a rule-based layer that suggests composting, recycling, or manual review. We first measure the frozen models on available development photographs, then examine how class-specific confidence thresholds affect their detection results. Finally, we audit the image and annotation records to determine the limits of those results.

## Related Work

Thung and Yang introduced TrashNet, an early approach to waste classification that assigns isolated objects to six material categories [2]. More recently, Single, Iranmanesh, and Raad developed RealWaste using photographs collected at a landfill, providing a setting with less controlled object appearances [3]. While these datasets support material recognition, they do not fully represent crowded cafeteria scenes in which several different items may appear together.

Proença and Simões addressed a related problem by introducing TACO, a dataset of litter photographed in context and annotated for instance segmentation [4]. Bashkirova and colleagues subsequently developed ZeroWaste, focusing on overlapping and deformable waste objects in recycling facilities [5]. These studies demonstrate the importance of detecting individual objects within cluttered images. However, the material or visual class of an object should not be confused with its correct disposal stream, which may vary according to local collection requirements.

Redmon and colleagues introduced YOLO as a unified approach to predicting object locations and classes [6]. This study uses Ultralytics YOLO11 [8] and reports average precision under the COCO evaluation conventions [7]. Although average precision summarizes performance over confidence-ranked predictions, it does not describe a system operating at one chosen threshold. We therefore report fixed-threshold precision, recall, and F1 alongside mAP.

Geifman and El-Yaniv studied selective classification, in which uncertain predictions can be rejected [9], and later incorporated a rejection option into model training through SelectiveNet [10]. Here, we use a simpler approach: choosing class-specific confidence thresholds for detectors that have already been trained. This allows us to evaluate the cost of retaining fewer predictions without changing the model weights.

Finally, the WILDS benchmark demonstrates the difficulty of maintaining model performance across different data distributions [11]. Kapoor and Narayanan similarly describe how data leakage and flaws in evaluation design can overstate machine-learning results [12]. Since the historical test photographs in this project were previously examined during development, the reported scores are treated as development results rather than independent estimates of cafeteria performance.

# Data

The maintained dataset contains 2,372 training views, 186 validation photographs, and 248 photographs in the historical test partition. Of the training views, 2,052 are original image entries and 320 are additional augmented views; twenty school photographs account for 340 views when their augmentations are included. Augmentation increases the variety of presentations seen during training without creating independently captured scenes. Because the specialist checkpoints were developed through different historical training and selection procedures, the current manifest should not be interpreted as the exact common training set for all four models.

The images come from public waste datasets and two school submissions. The historical cohort combines 73 school-lunch dataset images, 66 tetra-pak images, 48 trash-detection images, 31 plate images, 20 utensil images, four macworx images, and six photographs collected at the participating school. Thus, only 2.42% of the historical cohort consists of school-origin photographs. The separate dataset named school_lunch is not part of that school collection, and the cohort cannot be treated as a representative field sample from the cafeteria.

We score detections against the frozen, repaired annotation manifests rather than older label files that no longer reflect the accepted evaluation records. The historical cohort contains 924 annotated objects. A previous validation audit documented discrepancies in 109 older on-disk label files, including ten involving bottle annotations. None of these labels was changed for the present editorial revision; the recorded manifests and hashes remain the authoritative evaluation inputs.

Annotation quality also limits the interpretation of the scores. Historical annotations were produced or reviewed with AI assistance, and the available records do not establish comprehensive independent human verification. The school checklist, for example, contains 123 entries with blank human-review fields. Dense food scenes can also make the boundary between adjacent servings ambiguous. This uncertainty is particularly consequential for hinged containers, represented by just two validation annotations and four in the historical cohort.

**Table 1.** Object boxes by class (training views include augmentation; Hinge means hinged container).

| Class | Train | Val. | Test |
|:--|--:|--:|--:|
| Food | 2,501 | 343 | 433 |
| Plate | 722 | 110 | 123 |
| Utensil | 457 | 101 | 98 |
| Hinge* | 185 | 2 | 4 |
| Carton | 972 | 94 | 134 |
| Wrapper | 316 | 85 | 75 |
| Bottle | 293 | 42 | 57 |

# Methods

The system combines four YOLO11 specialists, each trained or adapted during earlier development to recognize a subset of the seven target classes (Table 2). The specialists' division of work was inherited from that development process; this study does not compare it with a single seven-class detector. Checkpoint hashes, class mappings, inference settings, and disposal rules were recorded in the frozen configuration at commit `defcc6156a4425daf1ae530757634cc541f3c267`. An integrity audit verified that the reported comparisons used the retained artifacts.

**Table 2.** Frozen specialist models. Input size is given in pixels; Hinge denotes hinged container.

| Model (input) | Parameters | Classes |
|:--|--:|:--|
| YOLO11n (640) | 2,590,425 | Food, hinge, carton |
| YOLO11n (640) | 2,605,126 | Plate, wrapper |
| YOLO11x (800) | 56,877,241 | Utensil |
| YOLO11s (640) | 9,428,179 | Bottle |

Inference was performed with Ultralytics 8.4.175, PyTorch 2.14.1, and Python 3.12. Each specialist processed one image at a time, without test-time augmentation, using its configured input resolution. Non-maximum suppression used an IoU threshold of 0.70 and a maximum of 300 detections per specialist. The saved protocol also retains half=False, an option deprecated in the installed library, to document the original invocation. Predicted boxes were represented in normalized XYXY coordinates. After each specialist's native suppression, outputs were concatenated by their assigned classes; no further cross-specialist deduplication was performed.

Evaluation took place on an Apple M1 MacBook Air with 8 GiB of memory and MPS support. These details identify the recorded computing environment rather than establish efficiency: latency, energy use, and throughput were not measured. Moreover, the four checkpoints have different adaptation and selection histories, so their combined performance should not be described as the outcome of a single controlled training experiment.

Detection outputs were retained down to confidence 0.001 for average-precision calculations. In the default display policy, predictions at or above 0.25 were shown, while scores below 0.50 were referred for review. At confidence 0.50 or above, food received a compost suggestion and bottles or drink cartons received a recycling suggestion. Plates, utensils, wrappers, and hinged containers were always referred for manual review. These are provisional rules rather than verified disposal decisions: the recycling suggestion does not establish that a container is empty or accepted locally, and the system does not issue automatic landfill recommendations.

![Processing pipeline: four YOLO11 specialists produce seven item classes, which are mapped to provisional compost, recycling, or manual-review suggestions.](paper/fig-pipeline.png){width=95%}

# Evaluation

We evaluate localization and classification jointly. Intersection-over-union (IoU) measures the overlap between a predicted bounding box and its reference annotation, divided by the area of their union. The primary ranking-based measures are mAP50, averaged across the seven classes at IoU 0.50, and mAP50–95, which additionally averages over thresholds from 0.50 to 0.95 in steps of 0.05 [7]. Both are calculated from confidence-ranked detections using the retained Ultralytics evaluation procedure; neither represents precision at one operational cutoff.

For fixed-threshold evaluation, detections are considered in descending confidence order and matched one-to-one to unused annotations of the same class when IoU is at least 0.50. Matched detections count as true positives (TP), unmatched detections as false positives (FP), and unmatched annotations as false negatives (FN). Aggregating these counts over all classes gives micro-averaged precision, recall, and F1:

$$
P=\frac{TP}{TP+FP},\qquad
R=\frac{TP}{TP+FN},\qquad
F_1=\frac{2PR}{P+R}.
$$

We distinguish these detection measures from two object-level proxies for automatic disposal suggestions. The first counts class-consistent matches separately for food, bottle, and drink carton. The second assigns displayed detections to reference boxes by geometry before checking their classes, thereby exposing competition between overlapping predictions from different specialists. A prediction that matches a food box but conflicts with a wrapper annotation illustrates why these approaches can yield different totals. Neither proxy can establish whether the object actually belongs in the suggested bin.

The evaluation also distinguishes development evidence from independent generalization. The 248-image historical cohort was examined during earlier model development and selection, so it cannot be treated as a sealed holdout. Class-specific thresholds were determined from the separate 186-image validation cohort and then applied unchanged to those historical images. Since thresholding only filters an existing ranked set of predictions, it changes fixed-cutoff precision and recall but does not change the frozen detectors' mAP. Previous use of evaluation data can otherwise inflate apparent performance [12].

# Threshold Selection

The initial experiment tested whether one confidence cutoff could improve precision while preserving useful detection coverage. On the 186-image validation cohort, we evaluated global thresholds from 0.50 to 0.99 at 0.01 intervals, together with 0.995. No single threshold simultaneously achieved the three prespecified criteria of at least 97% micro precision, at least 60% recall, and at least 100 accepted predictions. Among the global cutoffs satisfying the latter two requirements, the highest precision was 96.04% at a threshold of 0.79, with 68.73% recall.

We then allowed each class to have its own confidence threshold. Candidate cutoffs for each class were summarized by their validation-set true- and false-positive counts. A dynamic program combined these classwise choices, preserving the largest attainable number of true positives for each feasible false-positive total. The final threshold vector maximized correctly accepted detections subject to the same precision, recall, and accepted-prediction constraints:

\begin{equation*}
\begin{aligned}
\max_{\boldsymbol{\tau}}\quad & TP(\boldsymbol{\tau})\\
\mathrm{subject\ to}\quad & P(\boldsymbol{\tau})\geq0.97,\\
& R(\boldsymbol{\tau})\geq0.60,\\
& N_{\mathrm{pred}}\geq100.
\end{aligned}
\end{equation*}

**Table 3.** Validation-selected confidence cutoffs by class; Hinge denotes hinged container.

| Class | Cutoff | Class | Cutoff |
|:--|--:|:--|--:|
| Food | 0.79 | Plate | 0.83 |
| Utensil | 0.50 | Hinge* | 0.50 |
| Carton | 0.84 | Wrapper | 0.83 |
| Bottle | 0.91 | — | — |

The selected vector was subsequently applied without modification to the historical cohort. It was not used to replace the existing disposal-policy settings, and no additional cutoff selection was performed on those historical images. Because those images had already informed development, the resulting comparison remains descriptive rather than an independent estimate of field performance.

# Results

**Table 4.** Detection precision, recall, and AP50 on 248 development images (%); Hinge denotes hinged container.

| Class | Precision | Recall | AP50 |
|:--|--:|--:|--:|
| Food | 97.03% | 75.52% | 83.91% |
| Plate | 87.29% | 83.74% | 87.04% |
| Utensil | 90.91% | 71.43% | 86.06% |
| Hinge* | 80.00% | 100.00% | 99.50% |
| Carton | 99.23% | 96.27% | 97.55% |
| Wrapper | 89.23% | 77.33% | 85.13% |
| Bottle | 76.74% | 57.89% | 75.27% |

**Table 5.** Detection counts, F1 and AP50–95 for the same cohort. TP/FP/FN are object counts; Hinge denotes hinged container.

| Class | TP/FP/FN | F1 | AP50–95 |
|:--|:--|--:|--:|
| Food | 327/10/106 | 84.94% | 55.09% |
| Plate | 103/15/20 | 85.48% | 71.60% |
| Utensil | 70/7/28 | 80.00% | 61.26% |
| Hinge* | 4/1/0 | 88.89% | 91.17% |
| Carton | 129/1/5 | 97.73% | 74.28% |
| Wrapper | 58/7/17 | 82.86% | 57.64% |
| Bottle | 33/10/24 | 66.00% | 59.62% |

On the historical development cohort, the frozen system achieved 87.78% mAP50 and 67.24% mAP50–95, compared with 86.36% and 65.25% on validation. Results varied considerably by category (Tables 4 and 5). For example, drink cartons achieved 99.23% precision and 96.27% recall at confidence 0.50, whereas bottle recall was 57.89%. Hinged containers achieved 99.50% AP50, but only four such objects were annotated in the historical cohort, limiting the interpretation of that figure. Across all seven classes, the historical cohort yielded 93.42% micro precision, 78.35% recall, and 85.23% F1 at confidence 0.50.

**Table 6.** Automatic-suggestion object metrics. Neither matching method establishes correct-bin accuracy.

| Metric | Per class | Joint geometry |
|:--|--:|--:|
| Matches / 510 | 489 | 483 |
| Precision | 95.88% | 94.71% |
| Recall | 78.37% | 77.40% |
| F1 | 86.24% | 85.19% |

Class-specific matching counted 489 correct object detections among 510 automatic suggestions; geometry-first matching retained 483. Five food detections and one bottle detection were instead assigned to overlapping wrapper labels under the latter convention. These are disagreements in object assignment, not verified wrong-bin outcomes. The frozen 248-image predictions were reproduced in a separate inference replay, and the present analysis uses those saved outputs without retraining the models.

**Table 7.** Default sorting-policy outcomes. Denominators depend on detections versus labeled objects.

| Outcome | Count (rate) |
|:--|--:|
| Compost | 337/881 (38.25%) |
| Recycling | 173/881 (19.64%) |
| All automatic | 510/881 (57.89%) |
| Manual review | 371/881 (42.11%) |
| Auto matched (IoU) | 489/924 (52.92%) |
| Auto correct class | 483/924 (52.27%) |
| Reviewed objects | 286/924 (30.95%) |
| Missed objects | 149/924 (16.13%) |
| Unmatched auto | 21/510 (4.12%) |
| Unmatched review | 85/371 (22.91%) |

At the display threshold of 0.25, the default policy emitted 881 detections: 337 compost suggestions, 173 recycling suggestions, and 371 manual-review outcomes. Automatic suggestions therefore accounted for 57.89% of emitted detections. This denominator is narrower than the set of actual objects in the photographs, since undetected items generate no suggestion or review record. Indeed, exact-class automatic matches account for 483 of the 924 annotated objects (52.27%), a coverage-sensitive detection proxy rather than correct-bin accuracy. We do not attach conventional object-independent confidence intervals to these results: predictions are clustered within photographs, annotations contain unresolved uncertainty, and the historical cohort has prior development exposure.

## Precision–Recall Trade-off

**Table 8.** Seven-class detection at IoU ≥ 0.50. P, R and F1 are percentages.

| Cohort / cutoff | TP/FP/FN | P | R | F1 |
|:--|:--|--:|--:|--:|
| Val. 0.50 | 638/48/139 | 93.00 | 82.11 | 87.22 |
| Val. classwise | 555/17/222 | **97.03** | 71.43 | 82.28 |
| Hist. 0.50 | 724/51/200 | 93.42 | 78.35 | 85.23 |
| Hist. classwise | 630/18/294 | **97.22** | 68.18 | 80.15 |

![Precision and recall at uniform and class-specific thresholds, compared separately on validation and previously examined historical images.](paper/fig-precision.png){width=90%}

Class-specific thresholds reduced false positives on the historical cohort from 51 to 18, but also reduced true positives from 724 to 630. Accepted predictions fell from 775 to 648. Precision increased by 3.80 percentage points, while recall fell by 10.17 points and F1 decreased from 85.23% to 80.15%. Validation F1 similarly declined from 87.22% to 82.28%. Thus, class-specific thresholding produced more precise accepted detections at the cost of missing more objects. It did not improve mAP or the underlying detectors.


# Data Independence

The principal threat to independent evaluation is the possibility that photographs, closely related capture sessions, or earlier prediction results influenced model selection. We therefore examined the retained local inventory rather than inferring independence from folder names. The audit identified 102,806 readable image paths, 67,422 distinct filesystem inodes, and 14,379 byte-distinct images. Among 9,468 candidate identities, 3,842 were flagged for possible exposure, capture-group overlap, or duplication; 5,482 had unresolved historical provenance; and 144 original HEIC files could not be visually decoded in the audit environment. None could be positively certified as independent under the available records (Table 9), although this does not establish that every candidate was actually used to train a checkpoint.

To identify potentially repeated imagery, the audit compared file-level SHA-256 and decoded-pixel hashes, then screened near-duplicates using rotation- and reflection-aware comparisons of 64-bit difference hashes at a Hamming-distance threshold of five. The screening pool contained 4,751 images associated with training directories or prepared manifests. These checks provide conservative evidence of possible overlap, not exhaustive proof of model exposure: directory membership does not guarantee that a particular optimizer consumed an image, while nonmatching hashes cannot exclude related crops, viewpoints, or photographs taken in the same session.

**Table 9.** Provenance of 9,468 candidate image identities. Categories do not imply confirmed training exposure.

| Candidate status | Images |
|:--|--:|
| Exposure/group/duplicate flag | 3,842 |
| Unresolved provenance | 5,482 |
| HEIC decoding unavailable | 144 |
| Confirmed independent | 0 |

Both school submissions have development-related provenance. The earlier submission contained 123 photographs and the later one 21; each set shared recorded training or development capture groups. Four images from the October 29 submission entered the training collection, excluding that capture day from independent evaluation. Historical descriptions such as “sealed test” must therefore be interpreted alongside later inference receipts and model-use records. Overlap counts describe different audit indicators and cannot be summed as if they identified disjoint training exposures.

The audit separately examined annotation provenance, screening 2,398 JSON files together with additional CSV, XML, and YAML records. It did not identify a confirmed set of independently human-adjudicated, object-level disposal-bin labels. An annotation marked “reviewed” in an earlier workflow is not, by itself, evidence that a human established the object's material or appropriate bin. Forty-two unreadable image aliases and five malformed or inaccessible JSON records were also documented rather than silently repaired. The complete provenance summary and file identities are retained in the research-readiness report and external audit artifacts.

**Table 10.** Independent sorting outcomes remain unmeasured; required evidence is shown.

| Outcome | Required evidence |
|:--|:--|
| Bin precision/recall/F1 | Independent images, human bin labels |
| Correct-bin accuracy | Verified local bin eligibility |
| Automatic object coverage | Complete independent annotations |
| Field review rate | Sealed school cohort |

The absence of these metrics should not be interpreted as zero performance. They are undefined because the audited collection does not contain the independent disposal-label evidence needed to compute them.

# Error Analysis

We examined the frozen predictions for food and bottle, two classes that contribute to automatic disposal suggestions. A missed object was assigned a diagnostic category according to the strongest same-class overlap in the saved outputs: a sufficiently localized detection below confidence 0.50, a candidate overlapping at IoU 0.10–0.49, or no qualifying same-class overlap. False positives were similarly grouped by localization, duplication, competing class, or lack of a corresponding annotation. These categories are algorithmic descriptions rather than verified causal explanations. In particular, an apparent background false positive could reflect an incomplete reference annotation.

**Table 11.** Food and bottle error diagnostics at confidence 0.50. Categories are provisional.

| Error category | Food | Bottle |
|:--|--:|--:|
| Correct matches | 327 | 33 |
| FN: below threshold | 53 | 19 |
| FN: localization | 48 | 1 |
| FN: no overlap | 5 | 4 |
| FP: localization | 2 | 0 |
| FP: duplicate | 1 | 0 |
| FP: background/label | 7 | 9 |
| FP: class ambiguity | 0 | 1 |

Errors were concentrated in a few demanding photographs. One buffet scene accounted for 52 of 106 missed food objects (49.06%), while a single outdoor-trash scene accounted for 10 of 24 missed bottles (41.67%). The totals therefore reflect substantial within-image clustering rather than independent failures across dozens of different scenes. Both photographs were retained in the evaluation.

A targeted AI-assisted visual inspection considered 12 error cases from ten unique photographs. The examples included a sports-style bottle outside the intended disposable-container scope, a possible unannotated bottle, ambiguous cylindrical packaging, crowded food portions, and sealed food packaging recognized as food. These observations help identify plausible annotation and class-boundary problems, but cannot determine actual material, contents, or bin acceptance from the available previews. No annotations or disposal rules were modified.

A previously completed bottle-specialist trial also provided a negative experimental result. One additional epoch of classification-head training decreased validation AP50 from 82.47% to 77.95% and AP50–95 from 62.00% to 59.18%. The candidate failed its predefined validation criterion and was neither activated nor scored on the historical cohort. All main-paper results continue to use the frozen bottle checkpoint.

## Additional Image Evaluation

The unchanged detectors were also applied to 44 source-screened public photographs: nine from bottle sources, six from carton sources, fifteen from utensil sources, twelve from hinged-container sources, and two from wrapper sources. At confidence 0.25, the system emitted 149 predictions—twelve compost suggestions, 49 recycling suggestions, and 88 review outcomes—while three photographs produced no displayed detection. These are output counts, not measurements against independently verified object or disposal labels.

A preliminary source-category check found at least one detection of the expected source class in 9/9 bottle photographs, 5/6 carton photographs, 12/15 utensil photographs, 4/12 hinged-container photographs, and 0/2 wrapper photographs. These proportions are not class recall: the source categories do not provide exhaustive instance-level ground truth, and several photographs contain multiple objects. Visual examination suggested repeated confusion between open foam containers and plates, missed wrappers, and overlapping predictions in crowded bottle scenes. In one example, a foil meal tray was classified as a drink carton with confidence 0.617 and consequently received a recycling suggestion; the photograph alone does not establish its correct local disposal stream.

A separate, provisional comparison used approximate AI-generated boxes for 41 of the 44 photographs, excluding three crowded bottle scenes. The frozen V1 model ensemble achieved 46.25% mAP50 and 21.16% mAP50–95, while a developmental hinged-container V3 candidate achieved 44.78% and 21.47%, respectively. Because the annotations were not independently human verified and the photographs had already been inspected during development, neither comparison constitutes a valid independent benchmark. The V3 candidate was not incorporated into the frozen system.

Taken together, these observations illustrate plausible failure modes and the limits of the available annotation process, not demonstrated field accuracy or a validated improvement under distribution shift. Case-level notes are preserved in [the archived visual audit](https://github.com/cx-57/yolo-cafeteria-research/blob/archive/pre-cleanup-20261101/docs/exploratory-44-visual-audit.md). The source images require rights verification before reproduction in a publication.

# Limitations

The first limitation is evaluation independence. Although the model configuration and reported outputs are frozen, the historical cohort was consulted during earlier development, and the full lineage of every candidate photograph cannot be reconstructed. AI-assisted object annotations introduce further uncertainty, particularly in scenes where serving boundaries or similar forms of packaging are ambiguous. These factors weaken claims of generalization even when a benchmark's numerical precision appears high. The broader literature on distribution shift and leakage underscores why reused or mismatched evaluation data may overstate performance in a new setting [11, 12].

A second limitation is the gap between object recognition and sorting correctness. The policy does not establish whether bottles and cartons are empty, whether packaging contaminates food scraps, or whether the local waste contractor accepts a particular material. Nor does the system automatically recommend landfill disposal. Manual review is an abstention, not a successful sort, while undetected objects are omitted from the displayed review rate. No evidence was collected about contamination reduction, user compliance, inference latency, or energy consumption in a live school deployment. Public-dataset licensing and attribution must also be verified before any image redistribution.

An independent field study would require photographs collected in previously unused school sessions, with inclusion criteria and capture-group boundaries fixed before inference. Human annotators blinded to model predictions should identify each eligible object and document its actual disposal destination according to school-specific collection rules and, where necessary, material or contents evidence beyond the photograph. Disagreements and uncertain items should undergo recorded adjudication. The frozen checkpoints, thresholds, and decision rules could then be evaluated without further tuning, with class-specific detection metrics, correct-bin precision and recall, manual-review workload, and automatic coverage reported together. In addition to accuracy among accepted recommendations, the study should report the proportion of all eligible objects correctly assigned without review; otherwise, abstention could make a weak system appear artificially accurate. Uncertainty should be estimated across independent capture groups rather than treating objects from one image as independent observations.

# Conclusion

This study evaluated a four-specialist YOLO11 system for recognizing seven categories of cafeteria waste. On 248 previously examined development photographs, the system achieved 87.78% mAP50 and 67.24% mAP50–95. Class-specific confidence thresholds increased object-detection precision to 97.03% on validation and 97.22% on the historical cohort, but lowered recall and F1. These results demonstrate the trade-off between accepting fewer false detections and missing more objects.

A central limitation is that the available images do not form an independent school-based test set, and verified bin labels are unavailable. The next step is to collect new cafeteria photographs and evaluate the unchanged system against independently reviewed object and disposal annotations. Only then can its effectiveness as a waste-sorting assistant be established.

# Reproducibility

The [project repository](https://github.com/cx-57/yolo-cafeteria-research) retains the inference code, class mappings, disposal rules, evaluation scripts, and summary results. The recorded configuration in configs/research-freeze.json identifies the four checkpoint hashes and fixed inference settings; results/sorting-verification.json contains the retained detection results, and results/research-readiness.json summarizes the provenance audit. Evaluation conventions and label requirements are described in docs/sorting-evaluation.md.

The complete image data, checkpoint binaries, and detailed local audit artifacts are held outside Git in the project's accompanying asset directory. The working data manifest and frozen repaired annotation files are likewise external. As a result, the public source code alone does not reproduce native inference. Historical source and license declarations are retained with the assets, but attribution and redistribution permissions have not been independently audited; the school photographs are not presented as a released dataset.

The repository documents the commands for reproducing the file audit, running software tests, and evaluating newly collected annotations. These procedures do not retrain the frozen detectors or establish independent sorting accuracy; the latter requires a separately collected, human-verified evaluation cohort.

\ifdefined\balance\balance\fi

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

School food-services staff assisted with cafeteria context and image collection but did not validate model performance or disposal rules.
