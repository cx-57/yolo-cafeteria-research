# Gentle full-feature adaptation V29

Start from the same immutable V23 alpha0.5 specialist (2cd1efe2…2289),
not V27's fresh COCO initialization or a degraded V28 trained head. Use
exactly V28's verified training configuration and the unchanged 246/67
validation. Prior external additions 57/109 plus round four 17/20 remain separate
in the data audit, with cumulative 74/129.

Unfreeze all 9,428,163 feature/head parameters (only DFL fixed), preserve
all 81 BatchNorm running statistics, and use AdamW 0.000001. Physical batch 4,
nbs 16, 640 pixels, seed 43, mosaic 0.5, scale 0.5, 40-epoch schedule match V28;
first review phase is three epochs and is not a convergence claim.

This tests whether gentle feature adaptation from the strong specialist
helps where fresh full-model training and head-only adaptation have not.
It changes both trainability and learning rate, so it cannot isolate their
individual effects. Fixed BatchNorm addresses microbatch stability on this
Mac and differs from the author's large-batch recipe. No paper replication
or independent test performance is claimed.

Native validation uses NMS 0.7. End-of-phase checks reload the selected
checkpoint ordinary and TTA at 0.7, followed automatically by a fresh
ordinary640/batch 4/NMS0.5 check of the best trained checkpoint. That check
may promote only a genuinely higher saved-model score and records the exact
weights hash. Current incumbent remains protected until then.

Use continue_training.py for sustained execution: it checks V28's saved
result, then runs three-epoch phases with automatic fresh trained-model
checks until AP50 is at least 0.85 or the 40-epoch schedule is exhausted.
Every phase archives its model bytes and report; identical best snapshots
are reused. continuation.json records actual phase outcomes and honest
status. A below-target schedule end does not claim success.

Restart preflight checks any existing saved-model result before training,
so a verified target is archived without another update. An interrupted
phase lacking a matching report stops with verification_required; run
external-data-v7/validate_exact.py --run <this directory> --candidate trained
--ordinary, then restart the controller. Four continuation tests passed
after observed RED, including mismatched weights/settings/audit hashes,
native-score false success, successful restart, and incomplete-phase guards.

The sustained controller is now launched; no V29 accuracy result is claimed
until its first completed saved-model check.
