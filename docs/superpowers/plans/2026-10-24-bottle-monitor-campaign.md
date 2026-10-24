# Bottle continuation monitor campaign V34

> **For agentic workers:** execute inline with superpowers:executing-plans. User explicitly resumed bounded specialist experiments toward90% AP50 without overfitting; the separate full-training pause remains.

**Goal:** Seek one genuine saved checkpoint AP50>=0.90 and AP50–95>=0.64 while reducing further adaptive use of the original246-image benchmark.

**Architecture:** Start balancedV19 and V33’s reviewed data. Reserve12 newly acquired Open Images positive photos selected with seed1043 before measuring model scores; remove all their source exposures/groups from training. Add independently AI-reviewed generic Open Images negative scenes only to the monitor. Check source-byte/pixel/near-capture overlap. Choose weights and epochs using the monitor, then run the fixed benchmark once for the chosen candidate after the declared campaign. This is a continuation monitor, not an independent NHS test; possible generic pretraining exposure and small positive support remain limitations.

**Constraints:** Frozen246images/67boxes/labels/protocol never change. No validation-derived data selection. Monitor score90% is never called benchmark success. Fixed maximum9epochs,head-only,BNfixed,AdamW1e-5,batch4/nbs16,640,seed43,mosaic0,scale0.25,translate0.05. Retain best monitorAP50/AP95/latest plus initialization. Freshly reload monitor candidates at ordinary640,batch4,NMS0.5,noTTA. Predeclared monitor choice: largest AP50+AP50–95, initial included, tie prefers initial then earliest; final benchmark once on chosen trained weights. No blanket assertion that overfitting is impossible.

- [x] Review monitor negatives and screen all monitor sources against training/frozen benchmark. Hold ambiguous sources and all related training views; fail on unresolved overlap.
- [x] RED→GREEN tests: group-aware partition removes all exposures, rejects byte/pixel overlap, preserves source labels and train-only provenance; monitor criterion cannot report benchmark success or use benchmark during selection.
- [x] Build separate training/monitor artifacts with exact source hashes and coverage. Preserve V33 prepared dataset unchanged.
- [x] Implement fixed9epoch monitored training and monitor-only selection; benchmark evaluator reachable only once at end. Preserve optimizer/checkpoints and prior leaders.
- [ ] Verify, review and push. Launch bounded specialist campaign; full training remains paused.

Ruling: user’s no-overfit constraint takes precedence over the old per-epoch benchmark selection recipe. Monitor baseline measurements may inform reporting but never change this declared selection rule/budget. No new worktree because prior artifacts use absolute workspace paths; isolate V34 directory.

Ledger:3dHash-only source flags were visibly different photos and explicitly resolved before monitor measurements.237project/10focused checks pass. Independent review P2(report persisted before history) fixed with validated no-inference receipt recovery and lifecycle tests. Native checkpoint/cache storage avoids iCloud offloading.
