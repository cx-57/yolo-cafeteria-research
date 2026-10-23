# Expanded context training

Six epochs completed without improving the saved control. Best newly trained
native AP50 was 81.46%; independent TTA/NMS 0.5 evaluation was 78.35%.
See phase6-outcome.json. Adds 16 AI-reviewed LVIS photos with
29 source bottle boxes to the existing 41-photo / 80-box supplement.
Source image bytes, full source annotations, review reasons, and conservative
validation-overlap screening are recorded. One source reflection box is
excluded; scenes with missing soda-bottle boxes, uncertain materials or
possible related validation captures are held. Reviews are AI-assisted,
not human verification. No independent test performance is established.

Training has 1,108 unique photos / 1,651 exposures, five prior quality holds,
and no extra hard-negative repeats. Validation stays 246 photos / 67 boxes.
Use the prior successful head recipe: freeze=23, fixed BatchNorm running
statistics, AdamW lr=0.00003, mosaic=0.5, scale=0.5, 640 pixels, batch 4,
nbs=16, seed 42. Start from the verified best saved model and keep the
control separately. Build verifies source bytes, geometry, labels and counts.
