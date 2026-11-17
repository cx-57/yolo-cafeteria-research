# Blur robustness experiments — November 16, 2026

Status: no experimental candidate accepted or deployed. The neck-only training jobs were suspended when the user paused model work and requested a GitHub push. Partial training is not a completed result. The five-screenshot diagnostic completed.

At the fixed 60% confidence cutoff and IoU 0.50, the trained blur candidate improved development recall:

| Condition | Original recall | Blur candidate recall | Original precision | Blur candidate precision |
| --- | ---: | ---: | ---: | ---: |
| Clean | 65.38% | 67.44% | 95.49% | 91.45% |
| Gaussian blur | 35.65% | 55.73% | 94.22% | 91.74% |
| Downsampled | 52.51% | 61.39% | 95.55% | 92.44% |

The candidate failed preservation gates. A fixed halfway weight blend and a fixed agreement-based detection fusion also failed. Increasing input resolution to 960 was rejected. The original live model and 640 input size remain selected.

Development comprises 186 source photos with clean and two synthetic degraded views each. These exposed development measurements do not establish real-world generalization. User screenshots were excluded from training and model selection. A 225-source screened subset of the historical test partition was prepared but not inferred on in this experiment; that partition has historical evaluation exposure and is not a globally unseen test. The separate 16-photo wrapper final test remains unscored.

Annotation reconciliation matched all 186 benchmark source hashes and rows to the existing audited validation manifest, with zero differences. Earlier stale-label speculation in the chronological work log was disproved. Large resealable cartons were intentionally excluded by the single-serve-carton taxonomy; labels were not changed to improve scores.

A matched neck-only experiment was started, freezing backbone layers 0–9 and Detect layer 23 in both clean-control and blur-training arms. It is paused, incomplete, and has no acceptance result. Saved detection-head state must be verified before any later acceptance.

## Snapshot contents

`snapshots/` preserves local experiment scripts, recipes, plans, compact metrics, and chronological notes. `snapshot-receipt.json` records their original paths and hashes. These are historical source snapshots, not a turnkey relocated package: scripts originally ran under `runs/sorter/<experiment>/` with local external assets. Some recorded absolute paths refer to that environment. Reproduce with the same pinned inputs and review dependencies before execution.

Datasets, generated images, raw prediction dumps, checkpoints, environments, private screenshots and training logs remain local under the existing ignore policy. No confidence threshold was lowered, no source dataset/checkpoint was overwritten, and no live model was promoted.

## Screenshot diagnostic

At the unchanged 60% cutoff, both original and blur candidate missed the 8:40:49 image. Blurry bread confidence increased from 33.0% to 56.8%, still below cutoff; the clearer bread remained detected (91.5% original, 89.3% candidate). Neither correctly recognized the yellow wrappers. These screenshots were inference-only and did not drive model selection. Full prediction receipts are included without the image files.
