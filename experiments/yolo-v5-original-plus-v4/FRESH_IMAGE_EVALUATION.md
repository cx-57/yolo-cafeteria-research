**Latest label/model update:** `annotations-ai-v2` corrects public09 chopstick pairs to match existing V5 labels:124 total boxes. Previous-model fresh scores become44.61% mAP50 /30.19% mAP50–95 solely from that label correction. Selected carton640 policy leaves fresh scores unchanged; trained640 alternate scores45.53% /30.78% and is not selected from these photos. All labels remain AI-provisional. See [new trial results](V5_LABEL_REPAIR_RESULTS.md). Earlier126-box figures below are superseded, not model gains.

# Fresh public-photo evaluation — 2026-11-07

Tested the frozen 86.15% development candidate on **15 newly collected public photos**:12 school meals plus3 supplementary container/wooden-cutlery stress cases. No new local captures were available. Each cohort was frozen before its own inference; photos were not selected by successful predictions and none was added to training.

**Update: the 15 photos now have 126 provisional AI annotations and scored results.** These labels are not blind or human verified; historical/pretraining exposure remains unknown. This is not a certified independent field benchmark. Photos are newly collected for this session, not necessarily newly taken.

The current hybrid scores **44.07% mAP50 / 29.60% mAP50–95**, compared with **32.35% / 21.77%** for the original V5 parent, averaged over five supported classes. At confidence0.25 and IoU0.50, the hybrid has63 TP,19 FP and63 FN:76.83% precision and50.00% recall. Utensil AP50 changes from64.74% frozen V1 to68.61% refined V1. Carton AP50 is0% (only two positives), wrapper19.51%, food56.39%, plate75.86%. Hinge AP is N/A because no primary hinge could be confidently asserted.

Artifacts: `annotations-ai-v1/index.html` (annotation-only review), `labels/` (YOLO labels), `annotation-policy.json`, `annotations.json`, `label-freeze.json`, `results.json`, and `RESULTS.md`. Labels were drafted on original-image grids, checked on annotation-only overlays, then hashed before scoring. Original images, training labels and model weights remain unchanged. Food grouping, transparent-film boundaries and tiny/clipped objects remain uncertain; use these figures as provisional diagnostics.

## Results to inspect

- `index.html`: paired original-V5-parent versus current hybrid overlays for all 15 photos.
- `utensil-before-after.html`: frozen V1 versus fine-tuned V1 utensil-only overlays on the same 15.
- `reviewed-findings.json`: explicit visual observations and limits.

Positive examples: the current utensil model detects the fork and knife in public01, and avoids multiple original-V5 utensil false positives in public06. Regressions/gaps: public07’s real spoon drops from 0.86 confidence in frozen V1 to 0.24 after fine-tuning; both visible US-style milk cartons in public04/public06 are missed; wooden cutlery in public14 is missed by both V1 versions; the small tray in a whole-person image produces no displayed detections. Display confidence remains0.25 throughout; counts/confidences are not accuracy or recall.

## Next improvement priorities

1. **Training-label cleanup.** A confirmed visible spoon is missing from a training-negative label. `training-negative-review.html` and its JSON queue prepare69 existing school-lunch negatives for a focused, disjoint audit. No corrections applied. The documented class policy includes chopsticks; review visible chopsticks as utensils rather than excluding them.
2. **Material and scene coverage.** Add separate training-only examples of wooden/reflective cutlery, US-style milk cartons and realistic backgrounds, from photo sessions outside this evaluation set. The available V5 dataset contains 565 tetra_pak,377 school_lunch and1 trash_detection carton box. The selected carton detector is frozen V1; its complete historical training corpus is unresolved, so these counts do not establish its training distribution or the cause of its misses.
3. **Scale/framing.** Validate a tray-focused capture or small-object inference strategy on development data. Do not claim a manual crop as an automated detector improvement.
4. **Independent labels.** Collect or independently check exhaustive boxes on a separate future field cohort before reporting fresh mAP. Keep these15 images out of training and checkpoint selection.

Tested one fixed validation-only old/new utensil confidence-consensus rule:50/50 averaging on matched post-NMS boxes atIoU>=0.95, no new detections. It reduced utensil AP50 to68.57% and AP50-95 to43.70%, versus the 69.55%/45.86% incumbent, so it was rejected. It was not applied to the fresh cohort. `validation-utensil-consensus.json` records the result. No weights or deployed policy changed.

## Checks

- Exact, decoded-pixel and conservative perceptual matching against 16,119 unique available local project image contents, including V5 and older assets. All15 passed; no certification against unknown pretraining/cropped variants. Source roots/errors/closest matches and reusable reference index saved.
- Exact known-image parity before external inference:92 normalized detections reproduced from saved validation output. The known image is excluded from the external cohort.
- Preserved validation-style multi-label NMS,800px CPU, batch1, confidence floor0.001, NMS IoU0.7, max300 per native detector, existing class ownership and wrapper geometry policy.
- All primary/refiner/input/implementation/prediction hashes and complete model cohorts checked. Four new inference tests plus eight existing focused geometry/legacy tests passed; read-only code review found no important issue.


Full local galleries/source attribution/receipts are under `~/Desktop/yolo-cafeteria-research-assets/artifacts/fresh-image-evaluation-20261107/`. Reusable inference CLI: `scripts/v5_fresh_inference.py --candidate <candidate.json> --images <folder> --output <new-folder>`. It emits predictions with unknown labels and no accuracy claim. The verified selected model remains86.15%/58.44% on its original development cohort; that score does not apply to these public photos.
