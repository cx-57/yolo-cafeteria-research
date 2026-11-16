# V5 label repair and carton resolution results

Selected by unchanged226-image development checks: original frozen V1 carton branch projected to one class at640px. Other owners and wrapper geometry remain800px. No new-photo training in selected weights. The three-epoch carton adaptation remains an eligible alternate at640, but scored lower on development and was not selected.

| Metric | Previous800px carton policy | Selected640px carton policy |
|---|---:|---:|
|Development six-class mAP50|86.15%|86.30%|
|Development six-class mAP50–95|58.44%|59.76%|
|Development carton AP50|98.53%|99.42%|
|Development carton AP50–95|69.15%|77.08%|
|15-photo diagnostic five-class mAP50|44.61%|44.61%|
|15-photo diagnostic five-class mAP50–95|30.19%|30.19%|

Trained640 alternate on15 photos: 45.53% mAP50 / 30.78% mAP50–95. It was not selected using these diagnostics.

Fresh annotations are AI-provisional, not blind/human verified. Paired chopsticks follow the existing V5 convention:124 boxes, not superseded126. Saved-model baseline44.61 replaces44.07 due only to that label correction. Hinge lacks positive support and is unscored. Original test split and deployed system untouched. Historical/pretraining exposure remains unresolved.

Usable selected manifest: `verified-original-640/candidate.json`. Native one-image640 parity exact; full development metrics reproduced. `fresh-resolution-check/index.html` shows before/selected/alternate overlays. All trials, rejections, source licenses and correction manifests retained.


## Training-label audit and rejected trials

Reviewed69 school-lunch training images with no utensil labels:44 contain visible spoons or chopsticks. Corrected draft:74 added utensil boxes, grouping a chopstick pair as one instance to match inspected existing V5 positives, plus29 missing bread-film wrapper boxes. AI-provisional; original dataset untouched. The earlier118-utensil-box draft split pairs and is superseded, retained only as a rejected experiment.

| Trial | Class AP50 | Class AP50–95 | Decision |
|---|---:|---:|---|
|Utensil incorrect per-stick draft, one epoch|67.59%|42.16%|Rejected|
|Utensil consistent paired-stick draft, one epoch|67.77%|41.94%|Rejected|
|Wrapper repaired-label fine-tune, one epoch, fixed geometry|59.72%|31.10%|Rejected|
|Wrapper fixed confidence recovery, three formulas|71.09–71.34%|36.59–36.75%|Below1pp AP95 promotion gate|
|Carton geometry averaging|98.53%|68.25%|Rejected|
|Carton neck/head, three epochs,800px|97.04%|65.95%|Rejected|
|Original carton branch,640px|99.42%|77.08%|Selected|
|Adapted carton branch,640px|98.53%|74.99%|Eligible alternate; lower development score|

The carton pilot added4 independent training photos (7 cartons), repeated8 times per epoch alongside128 original replay images. These are4 new photos, not32. Two other photo families/3 cartons were checked after the800px decision; they remain diagnostics for subsequent resolution work. The selected original640 branch has zero new-photo fine-tuning. All checksums, splits, trials, annotation-only overlays, source/license metadata and model manifests are in `/Users/Charles1/Desktop/yolo-cafeteria-research-assets/artifacts/v5-label-repair-20261107`.

## Implementation checks

Inference supports explicit per-detector image size; current manifests override only carton to640. Other owners and wrapper geometry remain800. Missing overrides default800. Unsupported parameter overrides and invalid resolutions rejected. Recorded per-owner sizes in run receipts. Six inference tests passed after two new tests failed before implementation; all56 V5 tests passed. Read-only code review found no important execution issue. Native helper reproduced exact640px normalized detections on a known development image. Original deployed seven-class system untouched.
