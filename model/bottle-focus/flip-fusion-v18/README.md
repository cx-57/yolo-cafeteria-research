# Original plus horizontal-flip weighted box fusion

Predeclare one fixed configuration: original and horizontal flip at full
640-pixel resolution; per-view NMS IoU 0.5; Weighted Boxes Fusion IoU 0.55,
equal weights, confidence type avg and skip threshold 0.001. Use pinned
e222b49c… alpha 0.75 weights. Both image orientations are supported by the
training augmentation; this removes the right-angle views from v17.

Reuse the previously tested v8 same-resolution flip geometry and original
MIT upstream WBF code. Preserve the v17 implementation and results. Five
fresh tests must verify two-view confidence normalization and native validator
integration. Validation stays 246 photos / 67 boxes, no label/training changes,
no independent test result. No score claimed yet.

Fresh evaluation: **80.8864% AP50**. See `flip2-wbf640.json`. This did not beat the saved ordinary incumbent.
