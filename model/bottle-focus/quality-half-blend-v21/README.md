# Half quality-head blend, CPU evaluation

Predeclare the half/half point between the same two pinned sources as v19:
e222 ordinary incumbent and c97b quality-trained head. The quarter point
scored 84.0711% with ordinary inference. Frozen features and BatchNorm
running statistics must match exactly; retain both training-audit hashes.
Use ordinary 640 inference, batch 4, NMS 0.5 and CPU with two threads so
GPU training can continue separately. Device is explicit in the report.
The fixed development set remains 246 photos / 67 boxes. No score claimed;
no independent test performance. Fresh GPU verification is available if useful.

Fresh evaluation: **83.4546% AP50**. See `alpha0.5-ordinary640-iou0.5-cpu.json`. This did not beat the saved ordinary incumbent.
