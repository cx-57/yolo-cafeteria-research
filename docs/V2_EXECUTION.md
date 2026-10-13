# V2 execution checklist — October 12, 2026

User-authorized scope: audit all 3,096 originals, review the 1,364 food-box proposals and 60 ingredient conflicts, apply verified changes only to V2, add diverse class examples, freeze one split, and run three complete 80–100 epoch comparisons. Retain identifiable partially covered plates. Preserve V1 data and weights. An independent test set is optional; real NHS photos are unavailable.

## Phase 1 — Dataset

- [x] Record V1 dataset and checkpoint fingerprints.
- [ ] Structural audit of all 3,096 images, labels, source groups and duplicates.
- [ ] Visual screening of every image; record per-image decisions and inspect flagged objects at full resolution.
- [ ] Review all proposed LVIS food boxes and adjudicate 60 ingredient/portion conflicts.
- [ ] Save approved additions/replacements with provenance to V2 correction manifests.
- [ ] Source and fully annotate additional independent photographs for weak/absent classes, including paper boats.

## Phase 2 — Frozen V2

- [ ] Build a new reviewed V2 directory, preserving V1 and the earlier provisional V2 smoke dataset.
- [ ] Apply source/capture/duplicate grouping and class-aware split selection.
- [ ] Verify all 11 classes have train and validation positives; record support and source limitations.
- [ ] Freeze images, labels, class policy, split membership and SHA256 fingerprint.

## Phase 3 — Full comparisons

- [ ] Pretrained YOLO11n baseline: 80–100 epochs.
- [ ] Pretrained YOLO11n: same schedule, cls_pw=0.25.
- [ ] Pretrained YOLO11s: same schedule and dataset.
- [ ] Compare AP by class, mAP50–95, precision, recall, wrapper/bottle confusion, latency and memory; select the supported winner.

## Current evidence

V1 preservation: `data/audits/v2_review_2026-10-12/v1_preservation.json`.
Earlier V2 two-epoch run is an engineering smoke test, not a completed baseline.
Training comparisons remain gated on completion of dataset review and all-class validation coverage.
