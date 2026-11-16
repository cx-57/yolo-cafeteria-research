# Frozen V1 models on the V5 validation dataset

V1 was four frozen YOLO11 specialist detectors: food/container/carton, plate/wrapper, utensil, and bottle. Its historical 87.78% mAP50 came from a different, historically exposed test cohort. It is not directly comparable with V5's development metrics.

Evaluated the three relevant V1 detectors on the exact same 226-image, six-class V5 development validation cohort used for the current V5 candidate. Bottle is excluded from every score. All detectors use 800px CPU inference, batch one, confidence floor 0.001, NMS IoU 0.7 and max_det 300; exported boxes are scored identically. Original V1 defaults used 640px for its food and plate specialists, so this experiment standardizes resolution rather than reproducing that older deployment setting.

| Class | V1 AP50 | Current V5 AP50 | Combined AP50 | Combined owner |
|---|---:|---:|---:|---|
| Food | 71.95% | 86.82% | 86.82% | Trained V5 parent |
| Plate | 65.40% | 93.62% | 93.62% | Trained V5 parent |
| Utensil | 59.24% | 52.42% | 59.24% | Frozen V1 utensil |
| Hinged container | 84.08% | 98.36% | 98.36% | Trained V5 parent |
| Drink carton | 98.53% | 52.78% | 98.53% | Frozen V1 food/container/carton, carton output only |
| Wrapper | 42.99% | 64.29% | 64.29% | Trained V5 parent |
| **Overall mAP50** | **70.37%** | **74.72%** | **83.48%** | |
| **Overall mAP50-95** | **39.88%** | **47.71%** | **56.04%** | |

The V5 comparison is the accepted two-epoch utensil/carton candidate, not the earlier fresh five-epoch baseline. The combined candidate uses three detectors with exactly one prediction owner per class; it does not append duplicate parent predictions. Both selected V1 class owners pass the existing gate relative to their V5 incumbent: at least +1 percentage point AP50-95 and no AP50 regression greater than two points. No checkpoint was retrained or edited for this comparison. The existing inference-only contiguous-memory adapter preserves the legacy V1 utensil MaxPool3d operation.

Before scoring, checkpoint, source cohort, label and saved prediction receipts were verified. V1 and V5 standalone measurements were freshly reproduced from their validated predictions before combining them. The four retained V5 class scores remain exactly unchanged. Additional three-epoch V5 classifier refinements finished but none passed the AP50-95 acceptance gate; they do not replace the recorded winners.

Local artifacts under `~/Desktop/yolo-cafeteria-research-assets/artifacts/v5-specialist-trials-20261106/`:

- `v1-on-v5-common800/comparison.json`: full V1/V5 comparison, model hashes, native class ownership and evaluation reports.
- `verified-candidate-v1-v5/candidate.json`: complete candidate provenance, selection decisions and all per-class AP50/AP50-95 values.
- `verified-candidate-v1-v5/predictions.json` and `prediction-receipt.json`: merged predictions and SHA256 receipts.
- `verified-candidate-v1-v5/README.md` and `per-class.csv`: readable ownership chart and results.
- `evaluate-v1-on-v5.py` and `combine-verified-v1-v5.py`: local execution drivers using repository verification/scoring helpers.

The combined candidate was selected and measured on the same development cohort. Labels are provisional audit320 annotations, and inherited V1/V5 training exposure remains unresolved. These results do not establish independent test accuracy. No reserved test images were evaluated. Original weights and the deployed seven-class system remain unchanged; the new candidate manifest is not a legacy seven-class runner configuration.

## Subsequent development candidates

The preserved one-epoch unified fine-tune passed the unchanged gate for food and wrapper, while V1 utensil/carton and the parent V5 plate/container predictions remained selected. This four-detector candidate measured **84.43% mAP50 / 56.70% mAP50-95**; wrapper AP50 became **70.36%**, AP50-95 **34.43%**. Food AP50 declined only 0.34 points, within the two-point guard, while AP50-95 increased 2.68 points. Its complete manifest is in local `unified-refinement-vs-v1-v5/`.

A three-epoch wrapper neck/box/class trial improved AP50-95 to 35.98% but reduced AP50 to 65.26%, so it was rejected as a replacement. One predefined geometry-only combination then averaged agreed wrapper coordinates 50/50 (expert confidence >=0.25, IoU >=0.55, one-to-one matching), retaining all incumbent confidence/class/count/order and adding no predictions. It passed the same gate: wrapper **70.36% AP50 / 36.20% AP50-95**; full candidate **84.43% mAP50 / 56.99% mAP50-95**. Other class scores are unchanged. This increases the requirement to five detector passes, so the simpler four-model candidate remains preserved. Local `wrapper-geometry-trial/` records full hashes, settings, predictions and a copy of the tested source. Five invariant-focused tests pass; read-only review found no important issue.

These are repeated development-cohort selections, not independent improvements established on new data. A separate fixed1024px V1 utensil resolution trial is pending; no higher-resolution gain is claimed. Original checkpoints and deployment remain unchanged.

## Preserved V1 utensil fine-tune: accepted

Two completed CPU epochs fine-tuned only the actual V1 classification branches, keeping the native three-logit pooling head. The standard YAML reconstruction path was deliberately avoided because it would drop that custom head. Actual count: 1,221,513 trainable / 55,655,728 frozen parameters. Original weight files remained unchanged.

On the same full226-image800px validation cohort, utensil **AP50 improved59.24% → 67.33%**, and **AP50-95 improved37.17% → 44.52%**. The complete combined candidate improved **84.43% → 85.78% mAP50**, and **56.99% → 58.22% mAP50-95**. All five other class scores are exactly unchanged. This passes the original acceptance gate.

Fresh verification reproduced the baseline and final merged metrics; checked source/checkpoint/prediction hashes and class ownership; and compared saved model state tensors. Every non-classification tensor was identical to the frozen V1 source checkpoint. Only classification tensors changed, and all three MaxPool3d branches remain. Winning utensil checkpoint SHA256: `3ed7c3ce0f0ec57fe37513b7c9a1bc8f2e54541c389d9e13386ca80dd195676b`.

Local immutable snapshot `verified-utensil-6733/` contains the complete four-owner-model plus wrapper-geometry policy, merged predictions, SHA256 receipt, CSV and README. The live continuation is separate: two further epochs from this winning checkpoint, lower lr0=0.00002, same frozen scope/data/validation gate. No additional-epoch gain is claimed while pending. Development-only, repeated-selection and inherited-exposure limitations still apply; no reserved test or deployment change.

## Four-epoch utensil continuation

The second two-epoch phase passed: utensil **69.55% AP50 / 45.86% AP50-95**; full hybrid **86.15% mAP50 / 58.44% mAP50-95**. All other class scores stayed identical. The saved original V1 non-classification state tensors remain exactly unchanged. Immutable local snapshot: `verified-utensil-6955/`; winning utensil SHA256 `7ef95467e111e1a6c2844a431599193cd9cd94eabcc0152b1a276ab33eaac797`.

A separate two-epoch800px training continuation is running after a successful native MPS loss/backward preflight, with lower learning rate, preserved head, frozen non-classification weights and fixed BN. Its final selection remains a complete800px CPU evaluation versus the69.55% incumbent. No pending-trial improvement is included in the table above or claimed here. Repeated-selection/provisional-label/exposure limitations remain.

The800px MPS continuation subsequently stopped at its GPU memory cap after saving epoch1. Its second epoch is being resumed on CPU from that checkpoint, with optimizer/scheduler and original caps preserved; no restart from the earlier winner. Final CPU800px comparison is pending. The verified86.15%/58.44% candidate remains selected. A separate read-only seven-photo training-negative audit found one clear missing spoon annotation; candidate correction was recorded but no active labels changed. No gain is claimed from that finding alone.
