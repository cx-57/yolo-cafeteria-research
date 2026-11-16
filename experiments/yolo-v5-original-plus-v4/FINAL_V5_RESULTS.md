**2026-11-07 update:** selected carton branch at640px improves development results to **86.30% mAP50 /59.76% mAP50–95**. Other classes unchanged. Fixed15-photo diagnostic remains **44.61% /30.19%** using corrected124-box provisional labels; no external accuracy breakthrough. See [label-repair results](V5_LABEL_REPAIR_RESULTS.md). The prior snapshot below is retained as history.

# Final V5 development candidate

**Final selection: 86.15% mAP50 / 58.44% mAP50-95.** Training and evaluation are finished; no additional trials are queued. Bottle is excluded.

| Class | AP50 | AP50-95 | Owner |
|---|---:|---:|---|
| food | 86.48% | 59.05% | Refined V5 |
| plate | 93.62% | 76.60% | V5 parent |
| utensil | 69.55% | 45.86% | Fine-tuned V1 (4 epochs) |
| hinged_container | 98.36% | 63.78% | V5 parent |
| drink_carton | 98.53% | 69.15% | Frozen V1 carton output |
| wrapper | 70.36% | 36.20% | Refined V5 + box refiner |

## What is selected

The candidate uses four class-owning detectors plus one wrapper geometry detector. Exactly one detector owns each class; the wrapper auxiliary only refines agreed box coordinates and does not add detections or change confidence scores.

Utensil AP50 rose from the frozen V1 specialist’s 59.24% to **69.55%**, with AP50-95 rising from37.17% to **45.86%**, after four accepted classification-only fine-tuning epochs. The preserved V1 custom pooling head remains intact. Saved-state verification found no changed non-classification tensors. The original frozen checkpoints remain preserved.

The last additional800px two-epoch trial (first epoch MPS, second CPU after a memory-cap stop) reached68.32% utensil AP50 /44.41% AP50-95. It was **rejected**; the earlier69.55%/45.86% checkpoint stays selected. No further run was started.

## Files

All model assets remain local under `~/Desktop/yolo-cafeteria-research-assets/artifacts/v5-specialist-trials-20261106/`:

- `verified-utensil-6955/candidate.json`: exact owner checkpoints, SHA256s, native mappings, inference settings and wrapper geometry policy.
- `verified-utensil-6955/predictions.json` and `prediction-receipt.json`: verified full226-image output and hashes.
- `verified-utensil-6955/per-class.csv` and `README.md`: readable results.
- `v1-utensil-continuation/fit/weights/best.pt`: final selected utensil checkpoint.
- `utensil-negative-annotation-findings.json`: seven-photo training-only audit; one clear missing-spoon label candidate recorded, not applied.

Winning utensil checkpoint SHA256: `7ef95467e111e1a6c2844a431599193cd9cd94eabcc0152b1a276ab33eaac797`.

The manifest records the measured local research pipeline; the old seven-class deployed runner was not changed. Five detector passes carry extra inference cost, and throughput has not been benchmarked.

## Verification and limits

Final checks passed: all selected detector/geometry checkpoint hashes, implementation hash, prediction/candidate receipts, complete source dataset verification,226-image prediction count and final rejected-trial receipts. All eight focused tests passed (five geometry invariants, three legacy preservation/backward/serialization checks).

All reported metrics use the same226 V5 development validation images with800px CPU inference and identical exported-box scoring. These data have provisional audit320 annotations, inherited V1/V5 exposure is unresolved, and many candidates were selected on this same cohort. **These are not independent test-accuracy results.** No reserved test was evaluated. Hinged-container support is only12 objects. The newer corrected B187 archive was unavailable; original/active annotations were not silently changed.

A subsequent [15-photo public visual stress test](FRESH_IMAGE_EVALUATION.md) showed both improvements and real gaps, including missed cutlery and US-style cartons. The86.15% development score must not be read as fresh-image accuracy. No independently labeled fresh mAP was available, and the selected model was not changed based on this visual set.
