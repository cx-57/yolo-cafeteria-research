# Later-feature adaptation for the medium specialist

The next sustained feature-adaptation trial follows the
six-epoch V24 head-only phase, which fell short. Pin its immutable phase-six best
checkpoint and its fresh saved-model verification. Refuse missing or altered
source bytes. Confirm the actual loaded model is single-class YOLO11m.

Keep the exact same reviewed data, fixed validation and 640 pixels.
Use training batch 1 to limit feature-gradient memory, nbs 16 accumulation,
and ordinary evaluation batch 2 to match the preceding saved-model check.
Unfreeze from layer 10 and reduce AdamW learning rate tenfold to 0.00001;
BatchNorm running statistics remain fixed. The 84.0784% YOLO11s incumbent
remains protected. Added round-four candidates are not used in this trial.

Plan up to 100 total epochs, disabling mosaic in the final ten. Six-epoch
review phases retain optimizer state instead of restarting learning. Stop
on a target-crossing native score for a fresh saved-model check. This is
paper-informed feature adaptation, not an exact replication of a published
model or a claim of convergence after a short phase.
