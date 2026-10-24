# Bottle reviewed-data and neck adaptation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Execute inline under the user’s continuing authorization; no approval handoff.

**Goal:** Continue toward one saved bottle checkpoint with AP50 >=0.90 and AP50–95 >=0.64, without forcing results.

**Architecture:** Keep V30’s bounded trial and immutable metric policy. Stage additional LVIS batches, review full scenes and target crops, screen frozen validation overlap, then create V31 with reviewed full-image exposures. Initialize from balanced V19; train neck and head while keeping backbone parameters and all BatchNorm statistics fixed.

**Tech stack:** Existing Python3.12, Ultralytics8.4.175, PyTorch2.14.1, MPS, PIL/OpenCV audit helpers.

**Spec:** User’s joint metric target and latest explicit instructions to keep working and ignore README workflow. Agent choice: full-image reviewed data and neck adaptation rather than extending an unproductive head-only recipe.

## Global constraints

Frozen benchmark246 photos/67 boxes and fingerprint/protocol unchanged. No source label edits. AI visual review is not human verification. Native epoch scores never substitute for saved-model fresh verification. Ordinary640,batch4,NMS0.5,noTTA; keep exact joint threshold, immutable snapshots, nonregression promotion and Pareto retention. Only one MPS training process at a time. Preserve optimizer and completed phase artifacts. Terminal status of an individual trial is not completion of the overall user task.

## Review focus

Reject changed source bytes/annotations, duplicate accepted photos, held overlap records, missing annotation indices and invalid/out-of-bounds boxes. Keep separate training-only copies. Verify exact trainable layers and frozen BN live. Reuse six tested joint-policy lifecycle checks. Preserve earlier best models even if adaptation regresses.

## Tasks

- [x] Inspect V30 progress; epochs4–5 show no native improvement, epoch6 active.
- [x] Stage batches5–6; inspect full scenes and uncertain enlarged target boxes. Bind decisions to source hashes and screen hashes.
- [x] Test new admission on changed bytes, altered annotations, held overlaps and duplicate records; observe RED then implement minimal admission.
- [x] Build V31 full-image data from V30’s conservative source holds and admitted new photos, preserving all frozen validation bytes; verify source and exposure accounting.
- [x] Configure V31 from balanced88877...1518. AdamW1e-5, freeze11, fixed BN, batch4/nbs16,640,seed43,mosaic0.5,scale0.5,40epoch schedule in3epoch phases. Retain AP50/localization/joint/latest roles and fresh validation.
- [x] Run project and focused checks, obtain independent review, fix material findings, push concrete artifacts.
- [x] After V30’s current phase has a saved/fresh result, supersede it if still unhelpful; launch V31 and check live layer guards. Continue assessing actual joint metrics.

## Ledger

Ruling: use current working directory for experiment artifacts because the existing training process owns absolute paths here; isolate new work under a new experiment directory, without editing the active trainer.
Ruling: prior authorization and repeated “keep going, don’t stop” opt out of further planning questions. README workflow is not an execution requirement for this task.

Ruling: installed YOLO11 YAML includes C2PSA at backbone index10. Freeze11 preserves the complete backbone; a new failing guard test caught accidental index10 updates, then passed after correcting the boundary.
