# Bounded head interpolation comparison

Predeclare two additional convex head averages: alpha 0.25 and 0.75,
where 0 is the pinned edab V6 initialization and 1 is the pinned 082b V8
trained head. The existing alpha 0.5 checkpoint scored 83.6832% in ordinary
640-pixel inference with NMS IoU 0.5. Compare the two new points using the
same settings and unchanged 246-photo / 67-box development benchmark.

Frozen backbone tensors and all BatchNorm running statistics must match
exactly. Only changed head parameters may differ. Save each averaged model
as a normal single-model YOLO checkpoint and evaluate freshly reloaded
bytes. No independent test claim; this is development hyperparameter selection.
