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
    \renewcommand\subsection{\@startsection{subsection}{2}{\z@}{-1.2ex plus -.3ex minus -.1ex}{.4ex plus .1ex}{\normalfont\normalsize\bfseries\itshape}}
    \makeatother
---


\begin{abstract}
Automated cafeteria waste sorting involves two distinct tasks: identifying discarded objects and determining the disposal stream appropriate for each one. This study evaluates a prototype that combines four specialized YOLO11 detectors, covering seven cafeteria-item classes, with a rule-based policy that suggests composting, recycling, or manual review. The frozen models were evaluated on 186 validation photographs and a historically exposed cohort of 248 photographs containing 924 AI-assisted object annotations. On the latter cohort, the system achieved 87.78\% mAP50 and 67.24\% mAP50--95. At a uniform confidence threshold of 0.50, micro precision was 93.42\% and recall was 78.35\%. Selecting separate thresholds for each class on validation increased precision to 97.03\% on validation and 97.22\% on the historical cohort, while recall declined to 71.43\% and 68.18\%, respectively. Historical-cohort F1 fell from 85.23\% to 80.15\%. Only six historical images originated from the participating school, and the available records contain no independently verified disposal-bin annotations. The results therefore demonstrate how confidence selection changes recognition performance; they do not establish correct-bin accuracy or field generalization.
\end{abstract}

*Keywords:* object detection, YOLO11, cafeteria waste, selective prediction, recycling.

# Introduction

At the end of a school lunch period, discarded food, drink cartons, utensils, and packaging may be mixed together on a tray or in a waste container. Computer vision can help identify these objects, but identifying an item and deciding where to discard it are not equivalent problems. A plastic bottle might be recognizable from its shape while its eligibility for recycling still depends on its contents, contamination, and local collection rules. The U.S. Environmental Protection Agency recommends checking local recycling guidance and emptying and cleaning recyclable containers [1]. A system that detects a bottle correctly may therefore still recommend the wrong bin.

Object detection is a useful starting point because it locates individual items in scenes containing several kinds of waste. Its predictions, however, involve uncertainty. A low-confidence prediction may be incorrect, while rejecting it may leave a real item unidentified. This distinction becomes important when detection results are used to support disposal decisions: a system can increase the proportion of correct predictions simply by withholding more of them, even as the number of correctly identified objects decreases.

This study examines a prototype consisting of four specialized YOLO11 detectors and a separate rule-based disposal policy. It addresses three related questions: how well the retained detectors recognize seven cafeteria-item categories on available development images; whether selecting confidence thresholds by class improves precision without retraining; and what additional evidence is required before object detections can be interpreted as reliable waste-sorting decisions. The analysis uses frozen checkpoints and recorded predictions, allowing changes in decision thresholds to be evaluated independently of changes in the detectors themselves. Because the nominal test photographs have prior development exposure, the findings are presented as development results rather than prospective field performance.

## Related Work

Early approaches to waste recognition often classified the contents of an entire photograph. Thung and Yang's TrashNet project focused on individual waste items against comparatively simple backgrounds [2], while Single, Iranmanesh, and Raad introduced RealWaste using material photographed at a landfill [3]. These datasets illustrate how differences in image collection and object appearance affect the task being studied. Neither whole-image classification setting fully represents a cafeteria scene containing multiple overlapping items.

Localization datasets address a different aspect of waste recognition. Proença and Simões introduced TACO, which annotates litter instances within their surrounding environment [4]. Bashkirova and colleagues developed ZeroWaste to examine deformable and overlapping materials in industrial recycling imagery [5]. These studies are relevant to the crowded scenes encountered in waste collection, although recognizing an object or material remains different from verifying whether a particular school accepts it for recycling or composting.

The detector family used in this project originates from the YOLO framework introduced by Redmon and colleagues [6]. We use the YOLO11 implementation released by Ultralytics [8] and measure localization performance using the average-precision conventions associated with the COCO benchmark [7]. Average precision evaluates confidence-ranked detections over multiple operating points; it does not state how many correct and incorrect detections a system will retain after choosing a particular cutoff. For that reason, fixed-threshold precision and recall are also central to this study.

Selective classification provides a related framework for deciding when to withhold a prediction. Geifman and El-Yaniv examined confidence-based rejection [9] and subsequently introduced SelectiveNet, which jointly trains prediction and rejection components [10]. In contrast, the present work does not train a rejection network. It applies different confidence thresholds to existing detectors and measures the resulting precision–recall trade-off. Finally, research on distribution shift, including the WILDS benchmark [11], and on data leakage in scientific machine learning [12] highlights the importance of separating development decisions from independent evaluation. These concerns motivate our explicit audit of the historical image and annotation records.

# Data

The working manifest contains 2,372 training views, 186 validation photographs, and 248 photographs in a historical test partition. The training count includes 2,052 original image entries and 320 additional augmented views. Twenty original photographs submitted by the participating school contribute 340 training views when their recorded augmentations are included. Augmented views present modified versions of existing photographs; they are not independent observations. Moreover, the four specialist checkpoints have different training histories, so the current manifest should not be described as a single common dataset used to train all four models from scratch.

The images were assembled from several public datasets and school submissions. The historical 248-image partition contains 73 images from a dataset labeled `school_lunch`, 66 from `tetra_pak`, 48 from `trash_detection`, 31 from `plate`, 20 from `utensil`, four from `macworx`, and six from the participating school's own collection. Thus, only 6/248 images (2.42%) were collected at that school; the separately named `school_lunch` source does not belong to the school's submission. The historical sample is consequently dominated by public-source imagery rather than representative observations of the intended cafeteria environment.

Table 1 summarizes the class distribution. The 248-image partition contains 924 annotated objects, but the distribution is highly uneven: food accounts for 433 objects, whereas hinged containers account for only four. The validation partition likewise contains just two hinged-container annotations. Strong average precision for such a sparsely represented class should therefore be interpreted cautiously.

The evaluation uses the retained, repaired annotation manifests rather than superseded label copies. A previous audit documented discrepancies in 109 older validation-label files, including ten files with bottle-label differences. Annotations were produced or reviewed with AI assistance, and comprehensive independent human verification is not documented. Ambiguous serving boundaries, visually similar packaging, and incomplete labels may consequently affect the measured results. No labels were modified for the present manuscript analysis.

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

The prototype combines four YOLO11 specialists, with each checkpoint responsible for a specified subset of the seven target classes (Table 2). One model detects food, hinged containers, and drink cartons; another detects plates and wrappers; the remaining two detect utensils and bottles separately. These divisions reflect the previously developed system. The present experiment does not establish that four specialists outperform a unified seven-class model, since no controlled comparison of those architectures was performed. Checkpoint hashes, class mappings, and inference settings were recorded in the research freeze at commit `defcc6156a4425daf1ae530757634cc541f3c267`.

**Table 2.** Frozen specialist models. Input size is given in pixels; Hinge denotes hinged container.

| Model (input) | Parameters | Classes |
|:--|--:|:--|
| YOLO11n (640) | 2,590,425 | Food, hinge, carton |
| YOLO11n (640) | 2,605,126 | Plate, wrapper |
| YOLO11x (800) | 56,877,241 | Utensil |
| YOLO11s (640) | 9,428,179 | Bottle |

Inference was performed with Ultralytics 8.4.175, PyTorch 2.14.1, and Python 3.12. Each checkpoint processed the same image independently at its recorded resolution. Test-time augmentation was disabled; each specialist applied non-maximum suppression with an IoU cutoff of 0.70 and a limit of 300 detections. Outputs were then mapped to the seven shared class names and concatenated. Because no additional suppression was performed across specialists, overlapping predictions from different categories could remain in the combined output. This detail matters when one physical item resembles more than one target class.

The prototype includes a separate rule-based disposal layer rather than a learned bin classifier. At the default display threshold of 0.25, detections below confidence 0.50 are referred for review. Above 0.50, food detections receive a compost suggestion, while bottle and drink-carton detections receive a recycling suggestion. Plates, utensils, wrappers, and hinged containers are always marked for review. No category produces an automatic landfill suggestion. These rules are experimental: the system does not verify an item's material, whether a carton is empty, or the school's specific collection requirements. A suggestion therefore describes the rule applied to a recognized object, not independently established disposal eligibility.

The recorded evaluation environment was an Apple M1 MacBook Air with 8 GiB of memory and MPS acceleration available. Neither inference latency nor energy use was measured. The retained checkpoint and configuration hashes were checked during the research audit so that the results below refer to a fixed system rather than a new training run.

![Processing pipeline: four YOLO11 specialists produce seven item classes, which are mapped to provisional compost, recycling, or manual-review suggestions.](paper/fig-pipeline.png){width=95%}

# Evaluation

Object detection was measured using intersection-over-union (IoU), defined as the area shared by a predicted bounding box and its annotation divided by the area of their union. We report mAP50, which averages precision across classes at IoU 0.50, and mAP50–95, which additionally averages across IoU thresholds from 0.50 through 0.95 [7]. These ranking-based metrics are distinct from the precision of predictions retained at one chosen confidence threshold.

For the fixed-threshold comparisons, predictions were considered in descending confidence order and matched one-to-one to unused annotations of the same class at IoU at least 0.50. Each successful match counted as a true positive (TP); unmatched predictions and unmatched annotations counted as false positives (FP) and false negatives (FN), respectively. Micro precision, recall, and F1 pooled those counts across all seven classes:

$$
P=\frac{TP}{TP+FP},\qquad
R=\frac{TP}{TP+FN},\qquad
F_1=\frac{2PR}{P+R}.
$$

An additional evaluation concerns the disposal-suggestion layer. For food, bottle, and drink-carton detections, classwise matching counts whether an automatic suggestion corresponds to an annotation of the same object class. A more conservative geometry-first match instead assigns predictions to annotated boxes before checking class agreement, exposing competition between overlapping specialists. Both methods measure object-recognition proxies: neither can measure correct-bin sorting without verified disposal labels. We also distinguish precision among accepted predictions from coverage of all annotated objects, because missed detections generate neither suggestions nor review requests.

The 186-image validation partition was used to choose operating thresholds. The 248-image historical partition was scored afterward without retuning those thresholds, but its photographs had already influenced earlier model development. Accordingly, these observations are not an independent test of generalization [12]. Confidence filtering changes the set of accepted detections and thus fixed-threshold precision and recall; it does not improve the checkpoint weights or the underlying ranking-based mAP.

# Threshold Selection

We first evaluated whether a single confidence threshold could achieve high precision while retaining a useful number of detected objects. Global thresholds from 0.50 to 0.99, in increments of 0.01, together with 0.995, were evaluated on the 186-image validation partition. The selection criteria required at least 97% micro precision, 60% recall, and 100 accepted detections. No uniform cutoff met all three. Among the global thresholds satisfying the recall and detection-count requirements, the highest precision was 96.04% at confidence 0.79, with recall of 68.73%.

We next allowed the cutoff to vary by class. For each of the seven categories, candidate thresholds were summarized by the true and false detections they retained on validation. A dynamic program combined these possibilities, choosing the threshold vector that maximized accepted true positives while satisfying the same aggregate constraints:

\begin{equation*}
\begin{aligned}
\max_{\boldsymbol{\tau}}\quad & TP(\boldsymbol{\tau})\\
\mathrm{subject\ to}\quad & P(\boldsymbol{\tau})\geq0.97,\\
& R(\boldsymbol{\tau})\geq0.60,\\
& N_{\mathrm{pred}}\geq100.
\end{aligned}
\end{equation*}

The resulting values appear in Table 3. Bottle received the strictest cutoff, 0.91, while utensil and hinged-container thresholds remained at 0.50. This method optimizes the mix of accepted detections on the validation set; it does not calibrate confidence scores as probabilities of correctness or guarantee that those thresholds will transfer to other image sources.

**Table 3.** Validation-selected confidence cutoffs by class; Hinge denotes hinged container.

| Class | Cutoff | Class | Cutoff |
|:--|--:|:--|--:|
| Food | 0.79 | Plate | 0.83 |
| Utensil | 0.50 | Hinge* | 0.50 |
| Carton | 0.84 | Wrapper | 0.83 |
| Bottle | 0.91 | — | — |

The selected vector was subsequently applied without modification to the historical cohort. It was not used to replace the existing disposal-policy settings, and no additional cutoff selection was performed on those historical images. Because those images had already informed development, the resulting comparison remains descriptive rather than an independent estimate of field performance.

# Results

The frozen system achieved 87.78% mAP50 and 67.24% mAP50–95 on the exposed 248-image historical cohort. Corresponding validation scores were 86.36% and 65.25%. Tables 4 and 5 break down the historical results by class, with fixed-threshold precision, recall, and F1 evaluated at confidence 0.50 and IoU 0.50.

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

Performance differed substantially across classes. Drink cartons produced 129 true positives, one false positive, and five false negatives, giving 99.23% precision and 96.27% recall. Bottles were more difficult: 33 true positives, ten false positives, and 24 false negatives yielded 76.74% precision and 57.89% recall. The food detector was precise when it returned a detection (97.03%) but missed 106 of 433 annotated food objects. Although hinged containers reached 99.50% AP50, that figure is based on only four annotated objects and cannot establish robust performance for the class. Across all seven categories, the uniform confidence-0.50 policy yielded 724 true positives and 51 false positives, corresponding to 93.42% micro precision, 78.35% recall, and 85.23% F1.

The difference between class-level and pooled measures is important. mAP gives each category equal weight in its macro-average, while micro precision and recall are dominated by classes with more annotated objects. Neither summary alone describes what happens when an individual category, such as bottle or hinged container, is underrepresented or difficult to recognize.

**Table 6.** Automatic-suggestion object metrics. Neither matching method establishes correct-bin accuracy.

| Metric | Per class | Joint geometry |
|:--|--:|--:|
| Matches / 510 | 489 | 483 |
| Precision | 95.88% | 94.71% |
| Recall | 78.37% | 77.40% |
| F1 | 86.24% | 85.19% |

The disposal-suggestion layer uses a different operating policy from the classwise threshold experiment. Among its 510 automatic suggestions, conventional class-specific object matching identified 489 correct-class detections; a geometry-first assignment retained 483. The six disagreements involved five food detections and one bottle detection competing with wrapper annotations. This demonstrates that combining specialists without cross-class suppression can create ambiguous object assignments. It does not demonstrate six incorrect disposal-bin decisions, since no independently verified bin labels were available.

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

At the default display floor of 0.25, the pipeline produced 881 detections. Of these, 337 were provisionally assigned to compost, 173 to recycling, and 371 to manual review (Table 7). Thus, 57.89% of *emitted detections* received an automatic suggestion. This percentage is not coverage of actual waste items: the system cannot recommend or review objects it fails to detect. With all 924 annotated objects used as the denominator, only 483 objects (52.27%) were matched to automatic predictions of the correct visual class under geometry-first assignment. Even that figure is a recognition proxy, not evidence that those objects were sent to the correct disposal streams.

## Precision–Recall Trade-off

The class-specific thresholds were selected using the validation partition and applied unchanged to the previously examined historical cohort. Table 8 compares this policy with a uniform cutoff of 0.50. The selected thresholds achieved the target of at least 97% validation micro precision, but their performance must be interpreted alongside the objects they excluded.

**Table 8.** Seven-class detection at IoU ≥ 0.50. P, R and F1 are percentages.

| Cohort / cutoff | TP/FP/FN | P | R | F1 |
|:--|:--|--:|--:|--:|
| Val. 0.50 | 638/48/139 | 93.00 | 82.11 | 87.22 |
| Val. classwise | 555/17/222 | **97.03** | 71.43 | 82.28 |
| Hist. 0.50 | 724/51/200 | 93.42 | 78.35 | 85.23 |
| Hist. classwise | 630/18/294 | **97.22** | 68.18 | 80.15 |

![Precision and recall at uniform and class-specific thresholds, compared separately on validation and previously examined historical images.](paper/fig-precision.png){width=90%}

On validation, precision increased from 93.00% to 97.03%, while recall declined from 82.11% to 71.43%. The same pattern appeared on the historical cohort: precision rose from 93.42% to 97.22%, but recall fell from 78.35% to 68.18%. The historical false-positive count declined from 51 to 18, a reduction of 33, while true positives decreased from 724 to 630, a loss of 94 correct detections. Accepted predictions declined from 775 to 648 and F1 fell from 85.23% to 80.15%. Validation F1 likewise decreased from 87.22% to 82.28%.

These results show that the validation-selected policy is more conservative, not universally better. It produces fewer false detections among those accepted but leaves more annotated objects unrecognized. The optimization did not change the underlying checkpoint weights or mAP. Moreover, it has not replaced the default thresholds used to issue the disposal suggestions summarized in Table 7.

# Data Independence

A file labeled as a test image is not necessarily independent of model development. A photograph may have been viewed during error analysis, included in earlier inference, duplicated elsewhere, or taken in a session already represented in training data. We therefore audited the retained image inventory and its recorded histories rather than assuming that the historical test folder represented a new sampling event.

The local inventory contained 102,806 readable file paths, 67,422 distinct inodes, and 14,379 byte-distinct images. Among 9,468 candidate image identities, 3,842 carried potential exposure or overlap flags, 5,482 had insufficient provenance to establish independence, and 144 original HEIC photographs could not be visually screened. No candidate was *confirmed* independent under the available records. This classification does not mean that every image was used by an optimizer; it means the audit could not establish the separation required for a new holdout.

Exact checks used file-level SHA-256 and decoded-pixel hashes, supplemented by rotation- and reflection-aware comparisons of 64-bit difference hashes within a Hamming distance of five. The screening pool included 4,751 images associated with training directories or manifests. These checks identify possible duplication and overlap, but cannot rule out shared capture sessions or related views merely because their hashes differ.

**Table 9.** Provenance of 9,468 candidate image identities. Categories do not imply confirmed training exposure.

| Candidate status | Images |
|:--|--:|
| Exposure/group/duplicate flag | 3,842 |
| Unresolved provenance | 5,482 |
| HEIC decoding unavailable | 144 |
| Confirmed independent | 0 |

The school-origin photographs require similar caution. The earlier and later submissions contained 123 and 21 photographs, respectively, and both shared recorded development or training capture groups. Four images from an October 29 submission were added to the training collection, so that capture session cannot serve as a prospective test. Historical phrases such as “sealed test” are therefore superseded by the later provenance records.

The annotation audit examined 2,398 JSON files in addition to CSV, XML, and YAML records. It found no confirmed collection of independently human-adjudicated, object-level disposal-bin labels. A prior “reviewed” flag does not, on its own, establish a human judgment about material composition or local bin eligibility. The audit also preserved 42 unreadable image aliases and five inaccessible or malformed JSON records as documented limitations rather than silently excluding them. Full diagnostics and protected-file hashes are recorded in `results/research-readiness.json`.

**Table 10.** Independent sorting outcomes remain unmeasured; required evidence is shown.

| Outcome | Required evidence |
|:--|:--|
| Bin precision/recall/F1 | Independent images, human bin labels |
| Correct-bin accuracy | Verified local bin eligibility |
| Automatic object coverage | Complete independent annotations |
| Field review rate | Sealed school cohort |

The missing measurements in Table 10 are not zero scores. They are undefined because the independent ground-truth information needed to calculate them was not collected or could not be verified. In particular, a high object-detection precision cannot be converted into sorting accuracy by assigning each detected class to its provisional rule-based bin.

# Error Analysis

Food and bottle predictions were examined separately because both categories can trigger automatic disposal suggestions. Missed objects were grouped according to the strongest matching-class candidate: a sufficiently localized box below confidence 0.50, a box with only partial overlap (IoU 0.10–0.49), or no matching-class candidate. False positives were categorized by localization, duplication, competing categories, or possible reference-label errors. These categories describe model outputs; they do not establish the causes of individual failures.

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

Errors were concentrated in a few difficult scenes. One crowded buffet photograph accounted for 52 of 106 missed food objects (49.06%), while one outdoor-trash photograph accounted for ten of 24 missed bottles (41.67%). Thus, the error totals are not equivalent to failures across the same number of independent scenes. Both photographs remained in the reported evaluation.

The breakdown in Table 11 helps distinguish confidence problems from localization problems. Among food false negatives, 53 had a same-class prediction below confidence 0.50 and 48 had only partial localization overlap. Nineteen of the 24 missed bottles had a candidate below the confidence cutoff. Raising the threshold can therefore remove detections for objects that were already difficult to recognize.

A targeted AI-assisted examination considered 12 cases from ten photographs. Examples included a reusable sports bottle outside the intended disposable-container category, possibly incomplete bottle annotations, cylindrical packaging, crowded food servings, and wrapped food recognized as food. These observations suggest possible sources of confusion but cannot determine actual disposal eligibility. No annotation was changed.

A previously completed bottle-specialist experiment provided a negative result. One additional epoch of classification-head training reduced validation AP50 from 82.47% to 77.95% and AP50–95 from 62.00% to 59.18%. The candidate failed the retained validation acceptance criterion, was not activated, and was not scored on the historical cohort. The main results continue to use the frozen bottle checkpoint.

## Additional Image Evaluation

The frozen detectors were also applied to 44 public-source photographs: nine bottle, six carton, fifteen utensil, twelve hinged-container, and two wrapper source images. At confidence 0.25, the pipeline produced 149 detections, resulting in twelve provisional compost suggestions, 49 recycling suggestions, and 88 review outcomes. Three images produced no displayed detections. These are output counts rather than measures of accuracy because complete, independently verified object annotations are unavailable.

At least one source-category detection appeared in 9/9 bottle photographs, 5/6 carton photographs, 12/15 utensil photographs, 4/12 hinged-container photographs, and 0/2 wrapper photographs. These rates are not object recall: a photograph containing several items can satisfy the check when only one is detected. Visual inspection suggested recurring confusion between open foam containers and plates, difficulties recognizing wrappers, and overlapping predictions in crowded bottle scenes. For example, one foil meal tray was detected as a drink carton at confidence 0.617 and consequently received a recycling suggestion. The photograph does not establish which disposal stream should actually accept it.

A separate exploratory comparison scored 41 of these photographs using approximate AI-generated boxes, excluding three crowded bottle scenes. The frozen V1 model ensemble recorded 46.25% mAP50 and 21.16% mAP50–95, while an experimental hinged-container V3 candidate recorded 44.78% and 21.47%. Because the reference boxes were not independently human reviewed and the images had already been examined during development, these figures are not an independent benchmark or a valid basis for claiming a measured change in generalization. The V3 candidate was not incorporated into the frozen system.

The case-level record remains available in [the archived visual audit](https://github.com/cx-57/yolo-cafeteria-research/blob/archive/pre-cleanup-20261101/docs/exploratory-44-visual-audit.md). Reproduction of the images would require a separate review of their reuse rights.

# Discussion

The classwise results reveal an important limitation that a single aggregate score would conceal. Drink cartons were detected with high precision and recall, while bottles were missed more often and food detections were affected by dense scenes. This pattern suggests that the visual variability and context of discarded objects matter, but the available experiment cannot isolate the effects of training data, object appearance, or specialist architecture. In particular, no controlled comparison against a unified seven-class detector was performed.

The threshold experiment also shows why maximizing precision alone is insufficient. The class-specific policy reduced false positives, but excluded enough correct detections to lower recall and F1. For a sorting assistant, a missed item receives no automatic guidance; a correctly detected item, meanwhile, may still receive an inappropriate disposal suggestion if the rule is too broad. Thus, detection precision is only one component of a useful decision system.

The default policy issued 510 automatic suggestions among 881 emitted detections, yet only 483 of 924 annotated objects were matched to automatic predictions of the correct visual class under geometry-first assignment. That 52.27% quantity measures recognition coverage under the rule layer, not the fraction of items correctly sorted. Manual review is a legitimate way to defer an uncertain decision, but it cannot be counted as success. A future study must measure correct-bin assignments, missed items, and review workload together rather than reporting accuracy among accepted suggestions alone.

The present contribution is therefore a documented recognition system and a reproducible comparison of confidence-selection policies, not a demonstration of automated cafeteria disposal accuracy. The provenance audit places an additional boundary on interpretation: performance measured on previously examined development images cannot be assumed to transfer to new school photographs.

# Limitations

The first limitation is that the historical cohort cannot be treated as an independent test set. Some images or related capture groups influenced earlier development decisions, and complete checkpoint histories are unavailable. The annotations also lack comprehensive independent human verification. Sparse categories, particularly hinged containers, and clustered errors in crowded scenes further limit what the reported classwise metrics can establish.

A second limitation concerns the disposal task itself. The rule layer does not determine whether containers are empty, whether packaging contaminates food scraps, or whether a material is accepted by the school's waste contractor. It does not automatically recommend landfill disposal. Consequently, neither the recognition benchmark nor the exploratory public-image tests measure correct-bin accuracy, waste contamination reduction, or actual user behavior. Inference latency and energy consumption in a deployed setting were also not measured.

A prospective evaluation should use photographs collected during new school lunch sessions, with capture groups and eligibility criteria fixed before any model inference. Independent annotators should label all relevant objects and establish their correct disposal streams under documented school collection rules. When appearance is insufficient, annotators should use material or contents information obtained outside the photograph and record unresolved cases rather than guessing.

The frozen detectors, thresholds, and disposal policy could then be evaluated without adjustment. The results should report object-recognition metrics together with correct-bin precision and recall, automatic coverage of all eligible items, missed objects, incorrect suggestions, and manual-review workload. Reporting only performance on accepted recommendations would favor policies that abstain frequently. Uncertainty should be assessed across independent capture sessions, not by treating multiple objects in one image as unrelated observations.

# Conclusion

This study evaluated four specialized YOLO11 detectors for seven cafeteria-item classes. On an exposed historical development cohort of 248 photographs, the system achieved 87.78% mAP50 and 67.24% mAP50–95. Validation-selected class-specific confidence thresholds increased measured object-detection precision to 97.03% on validation and 97.22% on the historical cohort, but reduced recall and F1. The stricter policy therefore improved precision among accepted detections rather than overall object recognition.

The analysis also distinguishes visual recognition from disposal accuracy. The recorded annotations support object-detection measurements, but they do not establish whether the rule-based suggestions place items in the correct bins. New, independently collected school images and human-verified disposal labels are needed to evaluate that task. Until then, the results support a development-stage recognition prototype and a measured precision–recall trade-off, not a verified cafeteria sorting system.

# Reproducibility

The [project repository](https://github.com/cx-57/yolo-cafeteria-research) preserves the inference and evaluation code, class mappings, disposal rules, and summary results. The frozen configuration in `configs/research-freeze.json` records the checkpoint hashes and inference parameters. The corresponding recognition scores appear in `results/sorting-verification.json`, and the image-independence and annotation audit is preserved in `results/research-readiness.json`. The procedure for evaluating independently verified disposal labels is documented in `docs/sorting-evaluation.md`.

Complete image datasets, trained weights, repaired annotation files, and detailed historical audit artifacts are stored separately from the Git repository. Thus, the public source code alone is insufficient to reproduce the checkpoint inference. Dataset attribution and redistribution rights have not been independently verified, and school photographs are not released as part of the code. The preserved scripts support further verification when the external assets and a newly annotated field cohort are available.

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
