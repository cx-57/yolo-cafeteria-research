# Rotation-view weighted box fusion

Predeclare one fixed configuration: 0/90/180/270 degree full-resolution
views, ordinary per-view NMS IoU 0.5, then Weighted Boxes Fusion IoU 0.55,
equal view weights, confidence type avg, skip threshold 0.001. Absent views
reduce confidence; supporting views contribute confidence-weighted coordinates.
Use pinned e222b49c… alpha 0.75 weights (ordinary AP50 83.7287%).
Validation stays 246 photos / 67 boxes. Reuse the tested v15 rotation adapter.
Fusion uses unmodified MIT-licensed upstream NumPy code, bound in source.json.

Tests cover known-coordinate averaging, absent-view penalties, empty outputs,
clipping on non-square frames and native validator integration. No training
or label edits. No score claimed; no independent test result.

Fresh evaluation: **83.1193% AP50**. See `rotation4-wbf640.json`. This did not beat the saved ordinary incumbent.
