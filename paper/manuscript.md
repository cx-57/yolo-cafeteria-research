---
title: 'Cafeteria Waste-Sorting Assistance with Specialized YOLO11 Object Detectors'
subtitle: 'Detection, Selective Prediction, and the Limits of Development-Set Evaluation'
author: |
  **Charles Xie**  
  Natick High School · Natick, Massachusetts, USA
date: 'Research Preprint — November 1, 2026'
documentclass: article
papersize: letter
fontsize: 10pt
geometry: ['margin=0.86in', 'headheight=15pt']
colorlinks: true
linkcolor: blue
urlcolor: blue
toc: false
header-includes:
  - \usepackage{microtype}
  - \usepackage{booktabs}
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhf{}
  - \fancyhead[L]{\footnotesize Cafeteria Waste-Sorting Research}
  - \fancyhead[R]{\footnotesize Research Preprint}
  - \fancyfoot[C]{\footnotesize \thepage}
---


**Abstract.** Improper cafeteria waste separation motivates image-assisted identification, but recognizing an object does not establish its appropriate disposal stream. This study characterizes four frozen YOLO11 object detectors covering seven cafeteria-item classes, together with a rule-based compost/recycling/manual-review suggestion policy. The system achieved **87.78% mAP50** and **67.24% mAP50–95** on 248 historically exposed development photographs with 924 AI-assisted object annotations. At confidence 0.50, overall micro object-detection precision and recall were **93.42%** and **78.35%**. A class-specific confidence vector chosen only on 186 development-validation images increased overall precision there to **97.03%** (recall **71.43%**). Without retuning, the same vector yielded **97.22% precision and 68.18% recall** on the exposed 248-photo cohort. The greater precision was obtained without altering model weights, at the cost of 94 fewer accepted true-positive detections. An additional 41-photo public-image diagnostic with approximate AI labels gave the frozen baseline **46.25% mAP50**. The provenance audit found **no confirmed independent cohort** or human-verified object-level disposal-bin ground truth. These experiments document the utility and limitations of specialist recognition and selective confidence filtering; they **do not establish field sorting accuracy or reduced waste contamination**.

**Keywords:** YOLO11, object detection, cafeteria waste, selective prediction, precision--recall, evaluation leakage, annotation provenance.


# Introduction

Cafeteria disposal requires decisions about food scraps, packaging, utensils, bottles, and cartons in mixed, frequently occluded scenes. Current waste-vision research commonly tackles either waste-type classification or litter detection; the narrower question of *which school bin accepts a particular item* additionally depends on material, contamination, emptiness, and collection rules. The U.S. Environmental Protection Agency advises checking local recycling rules and notes that plastic, metal, and glass containers generally need to be empty and rinsed [6]. An object's detected class should therefore not be conflated with its certified disposal eligibility.

This paper studies an intermediate task: object-level cafeteria recognition with a deterministic *suggestion* layer and human-review abstention. Four separately adapted YOLO11 specialists recognize seven visual categories. Rather than presenting a new from-scratch training run or an automatically deployed sorting device, we preserve the original checkpoints and report their recorded performance, a class-specific confidence-policy experiment, and an unusually detailed provenance audit.

The central research questions are: **(RQ1)** What object-detection performance does the frozen seven-class system demonstrate on available development images? **(RQ2)** Can validation-selected, per-class confidence cutoffs improve overall object precision without retraining, and what fraction of objects is then missed? **(RQ3)** How do historical data exposure, annotation uncertainty, and missing bin labels bound the conclusions? The primary contribution is an auditable distinction among *recognition precision*, *decision coverage*, and *verified correct-bin accuracy*, quantities that are often inadvertently collapsed into one headline.


# Related Work

**Waste recognition.** Thung and Yang's TrashNet project classified isolated objects into six broad waste categories [1]. The TACO dataset of Proença and Simões instead treats litter in complex contexts with instance-level masks; its authors highlight the need for substantially more verified annotation to support deployment [2]. Mixed cafeteria scenes likewise demand multiple detections per photo, but visual labels such as “carton” and “food” are not directly interchangeable with collection eligibility.

**Detection evaluation.** COCO established widely used object-instance localization and precision--recall evaluation procedures [3]. Ultralytics YOLO11 supplies the underlying model family in this study [4]. We separate confidence-integrated class AP from operating-point micro precision: the former averages ranked detections, whereas the latter describes one particular confidence policy.

**Reject-option decision systems.** Selective classification explicitly trades prediction coverage for reduced error among predictions that are emitted [5]. This trade-off motivates evaluating *both* precision and recall when thresholding detections. In this project, manual review indicates that the policy declines to issue an automatic disposal suggestion; it does not count as a correctly sorted object.


# Data and Annotation Provenance

Public source manifests, source URLs and historical license declarations are
preserved externally in `data/datasets.csv` and `data/manifests/`. School originals
are preserved separately with SHA-256 identities, conversion records and capture
metadata. These records require source/attribution verification before public data
redistribution; this work did not independently audit licenses.

The working combined manifest is external
`data/combined-seven-school-oct29-20261031/manifest.json`. Its nominal train,
validation and test roles do not establish independence for the inherited active
checkpoints. The train partition contains 2,052 real photos and 320 additional
augmentation views, for 2,372 views. Twenty school training originals contribute
340 views including originals. Additional views do not add independent photographs.
Validation contains 186 real photos (seven school); test contains 248 (six school).
These are available working data, not a claim that each active specialist was
trained from scratch on exactly this single manifest.

Evaluation uses the frozen, embedded repaired annotations in
`artifacts/validation-label-audit-20261030/audited-eligible.json` and
`artifacts/test-label-audit-20261031/audited-all248.json`. The test manifest has
924 object boxes. The working dataset's older test rows are superseded for scoring;
do not use its older object totals. The validation manifest has corrections absent
from 109 referenced disk label files, including ten with bottle-label differences.
The files retain their recorded hashes. Embedded frozen rows, not those older disk
labels, define this evaluation. No labels were changed during this finalization.

Annotations are AI-assisted and retain uncertainty; historical words such as
"manual" or "verified" are not proof of human authorship. The school review CSV
has 123 blank `human_reviewed` entries. Food annotations aim at visually distinct
servings, but dense buffet examples expose possible serving-granularity ambiguity.
Only **two** hinged-container validation objects and **four** historical-test objects are available.
High scores for this class therefore provide very limited evidence.

**Table 1.** Annotation support by class. Training counts are augmented *views*, not independent photographs; validation and historical test counts use repaired object labels.

| Class | Training-view boxes | Validation boxes | Test boxes |
|---|---:|---:|---:|
| food | 2501 | 343 | 433 |
| plate | 722 | 110 | 123 |
| utensil | 457 | 101 | 98 |
| hinged_container | 185 | 2 | 4 |
| drink_carton | 972 | 94 | 134 |
| wrapper | 316 | 85 | 75 |
| bottle | 293 | 42 | 57 |



# Frozen System and Inference Protocol

The paper freeze is [research-freeze.json](../configs/research-freeze.json), rooted
at Git commit `defcc6156a4425daf1ae530757634cc541f3c267`. It pins all four checkpoint
hashes, specialist/rule files, benchmark summaries and inference profiles. The
analysis script checks these protected hashes before and after its work. This is a
recorded experimental freeze, not a filesystem lock or claim that all historical
assets have a complete integrity inventory. Future changes must be separate
experiments and must retain this freeze.

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

The recorded local environment is an Apple M1 MacBookAir10,1, eight CPU cores,
8 GiB RAM, with MPS available. This is environment documentation, not a latency,
throughput or energy benchmark. None of those efficiency claims was measured.
The active models have heterogeneous historical adaptation and selection histories;
the preserved recipes and release receipts should accompany any training narrative.
This pipeline is **not** one controlled, newly trained from-scratch experiment.

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


# Evaluation Design and Metrics

Detection AP uses the existing native Ultralytics evaluator, with mAP50 and the
mean over IoU 0.50:0.05:0.95. Fixed-threshold object precision/recall/F1 use
confidence-descending one-to-one IoU>=0.50 matches requiring an exact class.
These matching conventions and AP's score sweep are distinct measurements.

The automatic object proxy is reported two ways. Class-specific matching sums
food, bottle and carton matches separately. Joint geometry-first matching orders
all displayed detections by confidence and assigns each to at most one unused
label by IoU>=0.50 without consulting class/category; exact object class is checked
after assignment. This exposes competition between overlapping food/packaging
labels. Neither method establishes disposal eligibility.

For an eventual independently labeled bin evaluation, geometry matching must
remain independent of the desired bin. For each true bin b, precision is
TP_b/(TP_b+FP_b), recall is TP_b/(TP_b+FN_b), and F1 is
2*TP_b/(2*TP_b+FP_b+FN_b). Wrong-bin and unmatched automatic predictions contribute
false positives; reviews and missed true items remain false negatives for their
true bin. Undefined denominators must be reported as undefined, not zero accuracy.
Report both correct-bin accuracy among matched automatic decisions and the
coverage-sensitive fraction correct automatic/all labeled items. The former
excludes reviews, misses and unmatched predictions and must not appear alone.
Also report precision across all automatic predictions, object coverage, matched
review, missed objects, and unmatched predictions. This prevents abstention or
false-positive filtering from inflating apparent performance.


For a predicted box and ground-truth box, intersection-over-union (IoU) is the intersection area divided by the union area. At a fixed minimum IoU of 0.50, sorted, confidence-descending, class-consistent one-to-one matches yield micro precision, recall, and F1:

$$
P=\frac{TP}{TP+FP},\qquad R=\frac{TP}{TP+FN},\qquad F_1=\frac{2PR}{P+R}.
$$

Every comparison involving the 248-photo historical cohort remains a **development comparison** because the model lineage and earlier acceptance decisions exposed that cohort. The baseline AP computation uses the saved native prediction ranks down to confidence 0.001; threshold tuning changes only which detections are accepted at the operating point, not the frozen checkpoint or baseline AP.


# Validation-Selected Precision Policy

A uniform-confidence diagnostic searched thresholds 0.50--0.99 in steps of 0.01, plus 0.995, on the 186-photo repaired development-validation set. No global cutoff produced **at least 97% micro precision** while retaining **at least 60% recall** and **100 accepted predictions**. The best global threshold meeting the recall and prediction guards was 0.79, giving 96.04% precision and 68.73% recall.

A second, **class-specific** search enumerated each category's validation TP--FP trade-off. A dynamic program retained the largest true-positive count for each achievable aggregate false-positive count, then selected the configuration maximizing correctly detected objects among policies satisfying the same precision, recall, and detection-count constraints:

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

Only after the vector was fixed was it evaluated on the 248-photo *historically exposed* cohort; no threshold was reselected there. Its numerically successful 97% result is therefore not an independently validated field-performance claim, and these thresholds have not been deployed as the default disposal-suggestion rules.


# Frozen Development Results

**Table 4.** Per-class fixed-threshold object-recognition metrics on the exposed
248-photo cohort (confidence>=0.50, IoU>=0.50); AP uses the full frozen score range.
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

The macro mAP50/mAP50–95 is **87.78%/67.24%** on the 248-image development cohort
and **86.36%/65.25%** on the 186-image frozen validation cohort. All-class test
micro precision/recall/F1 at 0.50 is **93.42%/78.35%/85.23%**. Per-class validation
metrics are preserved in `results/sorting-verification.json` and
[the evaluation report](sorting-evaluation.md).

**Table 6.** Automatic-suggestion object proxies, explicitly separate from bin accuracy.
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

**Table 7.** Sorting output and coverage denominators on development images.

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

No automatic landfill suggestion is produced. The emitted-prediction review rate
does not include undetected real items. The 52.27% fraction measures exact-class
automatic matches, not correct disposal. No conventional image- or object-iid
confidence interval is supplied: objects are clustered within scenes/capture groups,
annotations are uncertain, and checkpoint/test exposure is present. A narrow
interval on these data would not establish independent generalization.



## Precision--Recall Trade-off After Threshold Selection

**Table 5.** Overall seven-class micro object recognition at box IoU $\geq$ 0.50; same source images and labels within each cohort.

| Cohort / policy | TP | FP | FN | Precision | Recall | F1 |
|:--|--:|--:|--:|--:|--:|--:|
| Validation (186), uniform 0.50 | 638 | 48 | 139 | **93.00%** | 82.11% | 87.22% |
| Validation (186), classwise selected | 555 | 17 | 222 | **97.03%** | 71.43% | 82.28% |
| Historical (248), uniform 0.50 | 724 | 51 | 200 | **93.42%** | 78.35% | 85.23% |
| Historical (248), classwise fixed | 630 | 18 | 294 | **97.22%** | 68.18% | 80.15% |

![Precision--recall trade-off for default and validation-selected classwise confidence thresholds, measured on exposed development cohorts.](paper/fig-precision.png){width=92%}

The historical cohort's accepted correct objects decreased by **94**, from 724 to 630, while false positives fell from 51 to 18. This is a precision--coverage trade-off, not a newly trained detector or improved mAP. Validation set precision rose by 4.03 percentage points; historical-set precision rose by 3.80 points. The selected thresholds produced 648 accepted historical predictions versus 775 previously. Further threshold selection on this historical cohort would make its evaluation even less independent.


# Independence and Human-Label Audit

The local inventory contains 102,806 readable image paths (67,422 distinct inodes),
which collapse to 14,379 byte-unique images. Aliases and repeated copies are not
independent samples. The original/candidate pool contains 9,468 byte identities
from raw/processed/public candidate sources and the direct school originals and
JPEG conversion directories. Review figures and experiment charts are excluded
from this candidate pool.

SHA-256 checks exact bytes; an EXIF-oriented RGB pixel hash checks identical decoded
pixels. Eight rotation/reflection versions of a 64-bit grayscale dHash screen
near-duplicates against 4,751 images with training-directory or prepared-manifest
membership, at Hamming radius five. These are review flags, not proof that two
photos are identical. Directory membership is potential exposure; it is not proof
that a particular active checkpoint's optimizer consumed every listed image.
Capture-group and recorded inference evidence are separately propagated.

The audit parsed 2,398 JSON files. The additional label search examined 207 CSV,
34 XML and 263 YAML files. XML objects were labeled `Plastic Bottle`, not disposal
bins; no per-object bin-label format was found in the JSON search. Historical
review/verified columns describe object review and do not establish independent
human bin truth. No verified human disposal-label set was available for scoring.

**Table 8.** Candidate provenance status. These categories do not assert that no
independent photo exists anywhere; they state what can be established locally.

| Status | Byte-unique candidate identities |
|---|---:|
| Exposure/group/duplicate flag | 3,842 |
| Unresolved historical provenance | 5,482 |
| Original HEIC visual decoding unavailable | 144 |
| **Confirmed independent** | **0** |

Reason counts overlap: 3,042 candidates share training-pool bytes/membership,
2,590 share a training-pool group, 2,682 have recorded development inference,
2,707 share a development group, and 588 trigger perceptual training-pool flags.
These are not additive counts of optimizer exposures. The HEIC files were byte
hashed; their 144 recorded converted school photos were decoded and all share
training/development capture groups. Conversion provenance is recorded, rather
than independently proved by a HEIC decoder in this environment.

All 123 school photos supplied in the earlier submission and all 21 in the newer
submission share training or development capture groups. The newer submission was
captured on October 29; four photos were admitted to training and its whole day
was excluded from independent testing. For the earlier submission, initial
"sealed test" or "no inference" flags are superseded by subsequent model-use
ledgers and prediction receipts. Historical test participation in release decisions
makes those data development evidence. A nominal split name cannot restore independence.

The audit also records 42 unreadable/dangling image aliases, one malformed JSON
process record, and four inaccessible historical JSON files. It did not modify or
repair any original. Unmatched public source photos remain unresolved because
complete lifetime fitting/selection histories and all capture relationships are
not available. dHash cannot rule out every crop, viewpoint change or adjacent frame.
Generic-pretraining exposure is also not certified absent.

**Table 9.** Independent real-world disposal metrics — unavailable, not zero.

| Metric | Independent result | Missing evidence |
|---|---|---|
| Precision, recall, F1 by actual bin | Not measured | Human per-object bin labels and independent images |
| Correct-bin accuracy | Not measured | Independently established disposal eligibility |
| Automatic coverage of independent objects | Not measured | Independently annotated object inventory |
| Manual-review rate on independent cohort | Not measured | A confirmed independent cohort run |



# Error Analysis and External-Image Stress Tests

The diagnostic decomposition uses frozen object labels and predictions, without
editing annotations or changing thresholds. A false-negative label is classified
as below threshold if a same-class box has IoU>=0.50 but score<0.50; a localization
candidate has same-class IoU>=0.10 but <0.50; otherwise it has no same-class overlap
at that cutoff and saved confidence floor. These are competing hypotheses, not
human-confirmed causes. A box can overlap other objects or depend on uncertain
serving definitions. Background-or-missing-annotation likewise does not distinguish
model hallucination from an incomplete label set.

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

One buffet image contributes **52 of 106 food misses (49.06%)**. One outdoor
trash scene contributes **10 of 24 bottle misses (41.67%)**. Object errors therefore
are strongly clustered; 106 or 24 misses are not that many independent failures.
Both difficult scenes remain in the primary recorded result.

AI visual inspection covered 12 deterministically selected error cases from 10
unique photographs; it is not a random audit or a human-label correction. Examples
suggest a reusable-looking sports bottle outside the disposable-bottle scope,
a possible missing bottle annotation, small cylindrical drink-package class
ambiguity, dense buffet scale/serving-granularity problems, and sealed food
packaging detected as food. Actual material, contents and bin acceptance cannot
be settled from these previews. No labels or sorting rules were changed.

The previously completed bottle experiment remains rejected. Its fixed one-epoch
classification-head trial changed validation bottle AP50/AP50–95 from
82.47%/62.00% to 77.95%/59.18%. Historical retention passed but the primary gate
failed. There was no candidate test scoring or activation. This is a documented
negative result, not a new model improvement. No retraining was performed here.

### Exploratory 44-photo public-source stress test (qualitative; not independent)

After the research freeze, the unchanged four detectors were run on 44
source-screened public photographs (nine bottle-source, six carton-source,
15 utensil-source, 12 hinged-container-source and two wrapper-source images).
There were 149 displayed predictions at the 0.25 confidence floor: 12 compost
suggestions, 49 recycling suggestions and 88 manual-review outcomes. Three photos
had no displayed detections. These are **prediction counts only**: there are no
human-verified object boxes, original capture-day independence certificates, or
school bin-eligibility labels, so recognition mAP, correct-bin accuracy and sorting
precision are **unmeasured** on this cohort.

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

These observations supply **qualitative failure-mode examples**, not evidence of a
measured performance change or an authorization to retrain on a future holdout.
Individual photo references and limitations are documented in
[the visual audit](https://github.com/cx-57/yolo-cafeteria-research/blob/archive/pre-cleanup-20261101/docs/exploratory-44-visual-audit.md). These public images may be
out-of-distribution relative to actual cafeteria disposal scenes, and original
source rights require review before reproducing them in a publication.



# Limitations and Future Evaluation

1. Development exposure and uncertain historical lineage prevent independent
   generalization claims for existing benchmark scores.
2. AI-assisted labels, small hinged-container support, and possible food-serving
   granularity and packaging class-boundary problems limit measurement validity.
3. Detection class is an insufficient proxy for material/reuse/contamination or
   disposal eligibility. The no-emptiness recycling shortcut may differ from
   actual school acceptance. It must be evaluated, not encoded into ground truth.
4. Review is abstention; omitted/missed items and unmatched predictions must remain
   visible in the results. No automatic landfill ability has been demonstrated.
5. The asset scan is bounded by readable local records and documented grouping.
   A no-match result is not a certificate of independence.
6. No new independent school capture session, human bin adjudication, latency,
   energy, or deployment-user study was completed. Source attribution/rights
   and the related-work bibliography need verification before submission.

To establish the missing real-world result, collect a genuinely unused cohort
from capture groups/days absent from training and selection, document its provenance,
and seal membership before inference. Include realistic mixed scenes and negatives;
define inclusion criteria before seeing predictions. Have human annotators, blind
to model outputs, label every eligible object with its box and actual school bin
using documented local acceptance/material/contents evidence. Record annotator
identity, time, adjudication and unresolved items; do not map detector classes to
"truth." Uncertain bins need explicit adjudication or a predeclared unscorable
status, with exclusions counted. Use multiple independent capture groups, determine
sample size from intended precision and bin prevalence, and report group-aware
uncertainty after evaluation. Do not choose thresholds using that holdout.



# Conclusion

The frozen four-specialist system produces seven classes of cafeteria-item detections with a historical development mAP50 of **87.78%** and mAP50--95 of **67.24%**. A validation-selected confidence vector achieved **97.03%** measured development-validation object precision and **97.22%** precision when fixed on an already-exposed historical cohort, at a substantial recall cost. These scores characterize object recognition, not correct compost or recycling decisions. Existing provenance and annotation uncertainty preclude claims of independent generalization or reduced contamination. A sealed, human-annotated school field trial with ground-truth disposal eligibility is the necessary next experiment.


# Reproducibility and Data Availability

Maintained code/configuration/results are in Git. Images, checkpoints, historical
experiments and the detailed 35 MB asset inventory remain external and intact.
GitHub alone therefore cannot reproduce native inference without the external
checkpoints and data. Research users should avoid imply that pushing the source distributed the weights.

- `configs/research-freeze.json`: checkpoint, policy, manifest and benchmark pins.
- `results/sorting-verification.json`: prior verified replay and detection metrics.
- `results/research-readiness.json`: this audit's compact findings and null bin metrics.
- `scripts/research_audit.py`: conservative inventory, pixel/perceptual screening,
  provenance indexing and tested error-diagnostic functions.
- External `artifacts/research-finalization-20261101/`: protocol, complete image and
  metadata inventory, read failures, error details, visual-review selection/notes,
  environment metadata and shareable statistical figure. Its summaries link hashes
  of detailed evidence; generated files are analysis artifacts, not new datasets.

From the repo root, with a fresh output path:

```sh
../yolo-cafeteria-research-assets/venv/bin/python -B scripts/research_audit.py   --output /tmp/cafeteria-asset-audit.json
../yolo-cafeteria-research-assets/venv/bin/python -B -m unittest discover -s tests -v
```

This audit does not run YOLO or train. Its dHash/source paths are defined in code;
future added files may change inventory totals without changing the frozen models.
The main replay/evaluation reproduction commands and frozen receipts are in
[sorting-evaluation.md](sorting-evaluation.md). Existing outputs refuse overwrite.
For actual independent bin metrics, use the existing `scripts/evaluate_sorting.py`
with new independently human-verified truth and the unchanged frozen sorting policy;
the required label format and matching convention are documented in that report.


# References

[1] G. Thung and M. Yang. *Classification of Trash for Recyclability Status*. Stanford CS229 Project Report, 2016. [Original report](https://cs229.stanford.edu/proj2016/report/ThungYang-ClassificationOfTrashForRecyclabilityStatus-report.pdf).

[2] P. F. Proença and P. Simões. *TACO: Trash Annotations in Context for Litter Detection*. arXiv:2003.06975, 2020. [Preprint](https://arxiv.org/abs/2003.06975).

[3] T.-Y. Lin et al. *Microsoft COCO: Common Objects in Context*. European Conference on Computer Vision, 2014. [DOI](https://doi.org/10.1007/978-3-319-10602-1_48).

[4] G. Jocher and J. Qiu. *Ultralytics YOLO11*. Software, 2024. [Documentation](https://docs.ultralytics.com/models/yolo11/).

[5] Y. Geifman and R. El-Yaniv. *Selective Classification for Deep Neural Networks*. Advances in Neural Information Processing Systems, 2017. [arXiv](https://arxiv.org/abs/1705.08500).

[6] U.S. Environmental Protection Agency. *Frequent Questions on Recycling*. [Official guidance](https://www.epa.gov/recycle/frequent-questions-recycling), accessed November 2026.

# Acknowledgments

School food services collaborated in providing cafeteria inventory context and facilitating school imagery. This does not imply that the school certified the model, its research evaluation, or actual disposal-stream eligibility.
