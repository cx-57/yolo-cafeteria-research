# Fixed quarter quality-head blend

Predeclare one parameter average: 0.75 of the ordinary 83.7287% incumbent
(e222…) plus 0.25 of the v11 quality-trained detection head (c97b…). The
latter used six additional TACO photos and reviewed negatives. Corresponding
frozen backbone tensors and BatchNorm running statistics must match exactly.
Retain both training-audit hashes. Evaluate freshly loaded normal single-model
weights with ordinary 640 inference, batch 4 and NMS 0.5. Fixed validation:
246 photos / 67 boxes. No score claimed; no independent test performance.
