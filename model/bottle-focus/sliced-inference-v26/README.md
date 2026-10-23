# Paper-informed sliced inference design

Goal: test whether enlarged original-resolution patches improve bottle AP50
on the fixed 246-photo / 67-box development validation set. User authorization
is the ongoing instruction to continue bottle experiments toward at least 85%.
This is a controlled research experiment, not an application workflow change.

Use the authors' SAHI 0.12.8 implementation, 384-pixel square slices with
25% overlap, retained full-image predictions, 640-pixel model input,
confidence 0.001, class-aware NMS with IoU 0.5, batch 1, max 300 final boxes.
Initialize the exact protected 84.0784% YOLO11s weights (SHA 2cd1efe2…2289).
Read original-resolution RGB photos; do not crop already-resized tensors.
Bridge predictions into the existing Ultralytics validator using its actual
letterbox ratios and padding. Verify known coordinates and real loader labels
before scoring. Preserve weights, image/label bytes and all split membership.
Report sliced inference explicitly, including its extra inference cost.
Never silently replace ordinary inference settings or claim an independent test.

Alternatives: longer feature adaptation remains prepared in V25; a larger
head-only model was tried in V24 and fell short. This inference-only comparison
is first because it directly targets our many small bottles and takes less
compute. No new source photos are required or admitted for this experiment.

Source: https://arxiv.org/abs/2202.06934 (Akyon et al., ICIP 2022).

Result: 76.3749% AP50, rejected. Matched ordinary MPS batch-one control
scored 83.4980%; the protected ordinary batch-four result stays 84.0784%.
All 246 images / 67 targets were evaluated, with 740 slices and 246 extra
full-image predictions. Evaluation wall time was 48.24 seconds, including
the native validator's unused ordinary forward pass for each photo.
