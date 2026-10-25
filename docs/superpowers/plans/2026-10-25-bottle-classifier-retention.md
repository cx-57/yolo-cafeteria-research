# Bottle classifier retention implementation plan

> Execute inline with superpowers:executing-plans; independent review under requesting-code-review. User authorizes optimization, caps full runs at20epochs; this trial is bounded at6epochs.

**Goal:** Address observed false-positive overconfidence while preserving the incumbent's decoded box geometry; require source-matched retention before another original benchmark assessment.

**Architecture:** Keep V42 immutable and restart from balanced88877bbf… . Twice-review the88training trash_detection photos, hold uncertain photos, correct the confirmed feather label only in a training derivative. Reserve a fixed-seed24-photo retention cohort (16positive/8negative) from training sources; remove all capture/byte/pixel/near aliases from the next training list. This is a retention diagnostic: the initial and V42 models have previously seen these photos, so it cannot establish unseen generalization.

**Tech stack:** Existing Python3.12/Ultralytics8.4.175/MPS runtime, unittest, Pillow/OpenCV.

**Spec/evidence:** V42 original benchmark79.397/57.973 vs incumbent84.071/62.061. Cached predictions only: at confidence.25/IoU.5 TP55→56, FP25→29; FP with confidence≥.5 8→21. Train classification loss.72254→.32648 while monitor classification loss2.45666→3.27111. Root found definite feather-positive training label. Source-monitor improvements failed to protect benchmark across V34,V36,V42; redesign monitoring rather than repeat its selection rule.

## Constraints / review focus

- Original246-photo/67-box benchmark, all old68monitor photos, old checkpoints and completed declarations remain unchanged.
- Never claim adjusting confidence changes AP or achieves90. No original benchmark epoch selection or threshold/TTA sweep.
- New retention photos/groups/aliases permanently excluded from future training. All source reviews are AI, not human certification.
- Only native YOLO11s classifier parameters `model.23.cv3.*` may update; all feature/neck/box parameters and all BatchNorm buffers stay fixed. Restore fixed EMA tensors after averaging to prevent numerical drift. Prove exact decoded box equality on random inputs before/after a real optimizer step and saved-checkpoint fixed-state equality.
- Max6epochs,b4/nbs16,640,MPS,AdamW1e-5,seed43,BNstatsfixed,mosaic0,scale.25,translate.05,warmup1,cosineLR,patience0. No unrelated recipe search.
- Training source/exposure counts derived from V42 after whole-photo holds, derivative feather correction, retention/group/overlap exclusions; do not quietly introduce new photos or duplicate supports.
- Evaluate exactly initial and V42 once on frozen retention cohort before new training, using640,b4,NMS.5,noTTA. This diagnostic does not use original benchmark inference.
- After epoch2, stop if either native retention metric drops more than.03below the saved initial retention result. Persist and honor stops/budget completion on restart. Native training validation uses retention cohort. Keep native bestAP50,bestAP95/latest (≤3distinct trained checkpoints). Freshly assess those on retention and old two cohorts. Eligibility requires retention AP50 and AP95 each ≥initial minus.005, both old source AP pairs nonregressing, and mean four-source gain≥.01. Eligible checkpoints rank minimum retention delta then mean other-source delta, tie initial then earliest.
- At most one gated final original benchmark candidate; promotion needs both benchmark metrics ≥84.071068%/62.060566%; move-on needs AP50≥90. No target guarantee.

## Tasks and interfaces

1. **Audit and reserve:** `data.py` consumes bound V42 audit/listing, original source manifest, bound root/second reviews. `prepare()->audit` creates immutable source identity/rows and seed20261025 selection before model measurement. `verify()->audit` recomputes source joins, held/alias exclusions, exposure counts, monitor copies/rows and hashes. Tests reject stale reviews, reserved groups/byte/pixel aliases and correction of any old source file.
   - [ ] Root88-photo review and independent whole-scene review; record explicit holds/correction.
   - [ ] Write failing partition/derivative-label tests; implement immutable source joins and training/retention outputs; run tests.
   - [ ] Screen retention against remaining training with existing SHA/pixels/rotation-dHash/SIFT conservative helpers; hold every flag and bind inputs.
2. **Protect geometry:** `geometry.py` provides `select(model)`, `fixed_state(model)`, `verify_fixed(model,reference)`, `restore_ema(ema,model)`. `trainer.py` freezes before optimizer construction, includes only classifier parameters in AdamW, fixes BatchNorm and EMA buffers, validates frozen state each epoch. Test actual YOLO11s random96forward/backward/optimizer/EMA roundtrip for changed classification scores with bit-identical decoded boxes and all fixed tensors.
   - [ ] RED→GREEN critical protection tests and wrong model/head fail-closed tests.
3. **Run bounded experiment:** `run.py` binds code/data/recipe/start checkpoint, freezes selection before measurement, diagnoses initial/V42 on retention, then trains at most6epochs and checks frozen saved weights. `finish.py` assesses up to3distinct candidates on declared cohorts and permits one original benchmark only after all gates. Report exact model/cohort hashes/support/settings and cached receipt identity on restart. Tests reject wrong weights/cohort/settings/nonfinite metrics and unsafe selection.
   - [ ] Declare, independently review code/data and verify benchmark/weights unchanged; run focused/project checks.
   - [ ] Launch authorized6epoch experiment, inspect each epoch, finish source-separated assessment, preserve outcome on GitHub; never replace a better incumbent with regression.
