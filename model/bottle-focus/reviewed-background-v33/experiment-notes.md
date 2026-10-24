# Bottle candidate prepared; full training paused

The user explicitly requested a pause before full training. No V33 training
has started and no training process is active. Further launch requires a new
user instruction. This directory stages a bottle specialist candidate; its
proposed head-only recipe has not produced any new accuracy measurement.

Best protected balanced pair remains84.0711% AP50 /62.0606% AP50–95;
legacy AP50 leader is84.0784% /60.9019%. V31 neck adaptation regressed:
fresh selected82.3376% /59.7363%,latest78.0905% /54.9265%,then archived
with optimizer retained. Exact90%/64% joint target is not achieved.

V33 training-only copy:579 distinct photos,538 source boxes,2148 scheduled
exposures. Retain all390 prior positive photos/1307 exposures; add23 fully
AI-reviewed Open Images photos/34 original boxes repeated4x. Every original
generic Bottle annotation in an admitted photo must be eligible. Whole-photo
holds include a missing fourth bottle (index8) and overlap screen(index25).
No source annotations, validation labels or splits were edited.

Review180 training background candidates:80 retained from84 legacy flags,
86 retained from a uniform seed43 sample of96 previously unknown absences.
Legacy flags alone were not trusted;166 photos passed full-scene root AI
and independent second AI review. Redistribute749 scheduled negative-image
exposures across these166;583 unreviewed/uncertain original backgrounds are
held for this experimental copy. Repeats change diversity and are not new
photos. This is AI visual assessment, not human/material certification.

Candidate recipe: balancedV19 initialization88877...1518;freeze23,all BN
fixed,AdamW1e-5,batch4/nbs16,640,seed43,mosaic0,scale0.25,translate0.05.
The planned data/geometry combination is not a causal ablation. Shared
fresh verification remains ordinary640,batch4,NMS0.5,noTTA,immutable
hash-bound checkpoints and exact same-checkpoint90%/64% thresholds.

All237 project and11 focused tests pass locally. Frozen246 image and label
hashes and both protected checkpoints were rechecked unchanged. GitHub CI
previously could not start because of account billing/spending limits;
local results must not be presented as a successful remote CI run.
The controller also requires512MiB free disk before a phase; current free
space is recorded in readiness.json and must be checked again at launch.
