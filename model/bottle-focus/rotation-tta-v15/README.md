# Full-resolution rotation augmentation

Predeclared inference trial: original, 90, 180 and 270 degree views of the
same 640-pixel input, all at scale 1.0. Invert each view's box coordinates
and concatenate unchanged confidence scores before ordinary NMS at IoU 0.5.
Use the pinned 42079f5a… averaged checkpoint. No training or label changes;
validation remains 246 photos / 67 boxes. Motivation: many litter bottles
are horizontal, and default scale/flip TTA does not test right-angle rotations.

Before scoring: known-coordinate tests on non-square frames must pass, plus
an actual-model probe preserving identity outputs, scores and model state.
A result will explicitly require four-view rotation inference and is not an
ordinary-inference score or independent NHS test result. Fresh saved-model evaluation reached 76.8113% AP50, with all 246
augmentation calls verified. This did not improve the 83.6832% ordinary
incumbent and was rejected. Both geometry/integrity tests and all 235 project
tests passed.
