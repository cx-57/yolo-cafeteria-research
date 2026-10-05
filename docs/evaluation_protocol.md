# Evaluation protocol

Status: draft. Freeze before final model testing.

Primary outcome: **disposal-guidance accuracy on an untouched NHS test set.**

Secondary outcomes:
- precision / recall / F1
- mAP50 and mAP50-95
- per-class AP and confusion matrix
- accuracy versus confidence threshold
- automatic-coverage / abstention tradeoff
- end-to-end latency
- robustness by clutter, occlusion, lighting, deformation, and contamination

Leakage controls:
- split by collection day / session / scene, not independent images
- never tune thresholds on the final test set
- never retrain on failed test examples and re-report on the same test set
- preserve an immutable final test manifest
