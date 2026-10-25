# Bottle reviewed full training implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task in this session.

**Goal:** Prepare a reproducible larger bottle training run seeking validated90% AP50 while retaining localization quality; pause before launching full training as the user requested.

**Architecture:** Preserve V36 training lineage and add only twice AI-reviewed, provenance-joined, overlap-screened V37/V38/V39/V41 sources outside both permanent monitors. Train all YOLO11s features from the protected balanced incumbent, with frozen BatchNorm statistics and a bounded20-epoch budget. Select using source-separated monitors; the original246-photo benchmark can evaluate at most one final candidate after monitor gates pass.

**Tech stack:** Python3.12, Ultralytics8.4.175, PyTorch MPS, Pillow, unittest.

**Spec:** User requests validated90% AP50 without overfitting, earlier64–70% AP50–95 aspiration, and pause before full training; V40 diagnostic and immutable source review receipts supply the data contract.

## Global constraints

- Full training requires the user to lift their earlier pause; preparation and verification proceed now.
- Balanced incumbent84.071068% /62.060566% remains protected; no promotion unless both original benchmark metrics are nonregressing for the same checkpoint.
- Old36-photo monitor and new32-photo litter diagnostic, source groups, byte/pixel aliases remain permanently excluded from training. Monitors are development evidence, not independent NHS tests.
- No source box edits, no synthetic crops, no new validation threshold/TTA sweeps. Source repeats are exposures, not new independent photos.
- Recipe: YOLO11s balanced88877bbf… init, all layers trainable except fixed DFL, BatchNorm running statistics fixed, max20epochs, patience10, AdamW1e-5, batch4/nbs16,640,MPS,workers0,seed43,mosaic.3,scale.5,translate.1,close_mosaic10,warmup1,cosineLR.
- Preserve2244 old exposures; new OpenImages twice, litter positives eight times, backgrounds four times. Expected725unique photos and2732exposures, final counts derived and verified.
- Combined68-photo monitor retains native bestAP50/bestAP95/latest; freshly evaluate at most3 distinct trained checkpoints on each separate cohort. Rank by minimum of four AP deltas against the initial model, tie by mean delta then earliest epoch. Require all four nonregressing and mean gain≥.01 before one original benchmark inference. Initial weights remain an eligible no-change outcome.
- Stop on insufficient disk before training (<2GiB), preserve optimizer/resume and selected checkpoints, never delete unique weights to free space.

## Review focus

- Stale review or annotation/metadata changes must fail source admission.
- Any reserved source/capture/byte/pixel alias must fail partition construction.
- Partial/stale dataset lists or changed label bytes must fail verification.
- Full training must refuse to start without explicit recorded user authorization bound to the declaration.
- Saved metric receipts must bind exact model, cohort, settings and support; nonfinite/regressing pairs cannot pass gates.

### Task1: Compile and reserve sources

**Files:** `model/bottle-focus/reviewed-full-training-v42/{sources.py,partition.py,build.py,test_partition.py}`.
**Interfaces:** `sources.compile_all()->dict(records,bindings,monitor_records)` joins raw ledgers and reviews. `partition.check(records,monitors)->None` rejects identity aliases. `build.verify()->dict` verifies exact dataset and immutable source hashes.

- [x] Write failing tests for reserved ID/SHA/pixel/group and duplicate source rejection; run unittest and confirm failures.
- [x] Implement source joins using V40 admission and V37 complete official groups, human plastic/water labels, photo metadata, root SHA and second review SHA. Reuse full V36 verification.
- [x] Build exact training exposure list and combined validation list outside iCloud; verify source and copied image/label bytes and immutable original benchmark.
- [x] Run focused tests and independent review; commit prepared data/audits.

### Task2: Prepare full training and monitor acceptance

**Files:** `model/bottle-focus/reviewed-full-training-v42/{protocol.py,test_protocol.py,run.py,declaration.json,experiment-notes.md}`.
**Interfaces:** `protocol.validate_authorization(receipt,declaration_sha)->None`; `protocol.rank(candidate,baseline)->tuple`; `run.py` checks immutable code/data/weights and authorization before calling training.

- [x] Write failing tests for missing/wrong declaration authorization and invalid/regressing monitor scores; confirm failures.
- [x] Implement the declared training recipe, actual backbone/neck/head trainability and BN checks, progress/checkpoint receipts, source-separated fresh evaluation and ranking. Do not launch.
- [x] Verify trainability without optimizer steps; run focused tests and root suite; independently review prepared source/runner.
- [ ] Preserve completed V36/V40 evidence and preparation on GitHub; present concrete ready run and ask to lift user's full-training pause. No elapsed-time approval.
