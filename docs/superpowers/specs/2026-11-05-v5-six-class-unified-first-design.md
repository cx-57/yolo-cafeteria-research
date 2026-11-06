# V5 six-class improvement experiments

Status: proposed experiment design; implementation and training have not started.

## Intent and user requirements

Remove bottle from V5. First improve the unified detector across its weak classes through multiple real training trials. Only if these trials fail to resolve a class's weakness, train and evaluate a specialist for that class. A specialist is conditional, not the predetermined outcome.

Retained class IDs stay unchanged: 0 food, 1 plate, 2 utensil, 3 hinged_container, 4 drink_carton, 5 wrapper. Bottle was ID 6. Historical V1 specialist weights, reports, and the uploaded seven-class V5 checkpoint remain preserved as provenance.

## Verified starting state

- Correct repository: https://github.com/cx-57/yolo-cafeteria-research (the supplied URL omits the second r in research).
- Inspected remote main at 8d94c0382edc56174862897e25861185b959da6e. Current working branch is an older field-generalization experiment, so implementation should start from current main on a separate branch.
- GitHub's V5 checkpoint metadata describes YOLO11s interrupted at 11/20 epochs, validation mAP50 0.65944 and mAP50-95 0.41966, without per-class results. Local Downloads/best.pt matches its SHA256: 62711f9deb1bb104eb23aabf5ae6fa83cba011c0aacee62877d04a77b827af27.
- These seven-class numbers are not a comparable six-class baseline. The checkpoint's exact balanced split has not been recovered locally; metadata alone cannot certify it. It must not initialize a new split if some proposed held-out images were in its training data.
- The old 97-image validation set has 274 food, 163 plate, 48 utensil and 13 wrapper labels, and zero hinged-container, carton or bottle labels. Reported utensil AP50 26.21% / recall 16.7% comes from that older limited split, not necessarily the 11-epoch checkpoint.
- Local Downloads/YOLO_V5_partial_annotation_audit320(5).zip contains 2,261 images (2093/97/71). Every manifest image and label hash verified. It is older than GitHub's latest documented YOLO_V5_CommandCenter_B187_QC_selected_merged1572.zip. No latest complete archive was found in Downloads or the project assets.
- Latest documented annotations remain provisional, with 142 flagged images, 28 especially ambiguous images recommended for exclusion, and unresolved V4 cross-class completeness. A mechanical hash check is not semantic annotation verification.
- Existing Python environment provides Ultralytics 8.4.175, PyTorch 2.14.1, and available Apple MPS; CUDA is unavailable. The checkpoint records Ultralytics 8.4.203. Record/pin the actual experiment environment rather than silently changing dependencies.

## Approaches and selected order

1. **Unified first, specialists only after measured failure:** the user's selected approach. It permits simpler inference if targeted training succeeds, and requires genuine comparable trial results before adding models.
2. **Immediate specialist ensemble:** could address a weak class sooner but conflicts with the requested order; rejected.
3. **Threshold tuning alone:** useful for choosing an operating precision/recall point, but does not establish an AP improvement; retained as evaluation, not counted as a training trial.

The concrete trial budget and numeric gates below are proposed implementation choices, not targets supplied by the user.

## Six-class dataset and fixed evaluation

Use the latest corrected complete archive once available. Do not silently substitute audit320 for it. If the user chooses the older archive, record that choice and its limitations explicitly.

Create a separate derived six-class dataset; preserve image bytes, remove only class-6 annotation rows, and set nc/names to the six retained classes. Retain bottle-only images as valid background examples, provided no retained-class objects are missing labels. Never delete an entire mixed-class image because it contains bottle. Record source and derived hashes, removed box counts by split, valid empty labels and before/after class counts. This changes the training task; a seven-output checkpoint with bottle filtered during prediction does not meet the requirement.

Freeze one group-aware approximately 80/10/10 split after the conservative ambiguity exclusions. Preserve historical held-out images outside training; keep exact duplicates and conservative near duplicates together; honor real meal/tray/session metadata where present. Dataset-wide source names such as `group=school_lunch` must not be mistaken for meal/session IDs. Do not silently ignore large genuine groups to reach target sizes. Check class support in train/val/test and disclose rare-class image and group counts. If safe split construction fails, report why rather than training on a leaky fallback. Save one immutable manifest for all candidates.

Audit labels in training and validation for weak-class omissions, with overlays and an explicit unresolved queue. Exclude the 28 unresolved ambiguous images if their identities can be established from the latest archive/ledger; fail the default preparation if the required ledger is missing. Do not silently convert uncertainty into confident labels. Freeze dataset/label hashes before the comparative trials; a later label change starts a new baseline/experiment family.

The expanded held-out cohorts remain development data: older models and annotation work may have exposed them. Train fresh from a general pretrained checkpoint for the new split. Do not claim a newly constructed split makes previously used images independent. Keep test metrics out of trial selection and evaluate the final selected pipeline once after selection.

## Unified trials

Start with a freshly pretrained six-class YOLO11s baseline on the fixed split. Use seed 57, image size 800, batch 4, AdamW, lr0 0.0005, cosine schedule, and the existing balanced-notebook augmentation recipe. Complete a 40-epoch baseline budget with no early stop before epoch 20; record actual completed epochs and validation-best checkpoint epoch. Longer convergence is necessary to distinguish undertraining from a persistent class deficit; the original 11-epoch run was interrupted.

Then run six controlled variants, each on the same labels/split and with the same seed and 40-epoch maximum budget. Each variant starts from general pretrained weights, not from the winner of another variant, so changes remain interpretable:

| Trial | Change from baseline | Question |
| --- | --- | --- |
| U1 | Reduce aggressive scale/mosaic augmentation; disable mosaic in the final 10 epochs | Does augmentation erase thin or small objects? |
| U2 | Increase training and validation image size to 1024; lower batch if needed and record effective batch | Does additional resolution improve weak-class localization/recall? |
| U3 | Capped deterministic oversampling of training images containing baseline-weak classes | Does training exposure improve rare classes without duplicating held-out images? |
| U4 | Lower lr0 to 0.0002, same optimizer and schedule | Does a gentler schedule improve localization and convergence? |
| U5 | YOLO11m pretrained backbone, batch adjusted for memory | Does extra capacity help at acceptable runtime cost? |
| U6 | Combine only the changes supported by U1-U5; if none help, extend the unmodified baseline schedule to 80 epochs | Do supported changes work together, or does the baseline simply need more convergence? |

Benchmark feasibility before committing to all runs. Memory failures, interrupted runs, smoke runs, and threshold sweeps are not successful completed trials and do not satisfy the specialist gate. Save resumable checkpoints, per-run environment/configuration, source and split hashes, training logs and wall-clock timings. Resume only when checkpoint/dataset/run configuration matches. Existing output directories must never be overwritten or used to select an unrelated newest checkpoint.

Evaluate every candidate on validation using AP50, AP50-95, per-class precision/recall curves, and image-level error overlays. Report recall at a validation-selected precision of at least 0.80; if unattainable, report it as unattainable. Use each class's operating threshold consistently when comparing its recall. Report latency and memory separately. Do not equate threshold changes with mAP changes.

## Proposed specialist gate

Define initially weak classes from the newly measured six-class baseline: AP50 below 0.80 or recall below 0.70 at precision at least 0.80. Treat inadequate validation group support as insufficient evidence rather than automatically selecting a specialist. Record support and image/group-bootstrap uncertainty, acknowledging that tiny cohorts may prevent a confident decision.

Complete the baseline and all six unified variants before opening the specialist stage. Re-run the baseline and chosen unified candidate at seeds 58 and 59 for three-seed comparisons. A class is eligible only if it remains below target and the best unified candidate fails to improve it reliably: proposed minimum useful changes are +0.02 AP50-95 or +0.05 recall at precision at least 0.80. Report both per-seed variability and paired group-bootstrap intervals; do not label a noisy delta as a reliable improvement. If meaningful progress is still occurring, continue the unified path rather than calling it exhausted. If uncertainty prevents a decision, collect/adjudicate additional validation labels before declaring failure.

Choose the unified candidate by weak-class AP50-95 improvement while limiting any other class's AP50-95 regression to 0.02. Record all per-class tradeoffs; if no candidate meets this guard, retain the baseline and report that outcome. The gate is a reproducible rule enforced from run receipts, not a notebook comment bypassable by jumping to the last cell.

## Conditional specialists

For each eligible class, train a class-specific model using only the fixed training pool. Keep training negatives, and remove non-owned class rows only in a separate specialist label view. Verify target-label completeness rather than introducing incorrect negatives. Keep validation/test ownership and image membership unchanged.

The unified model retains ownership of classes that do not qualify. An accepted specialist replaces the unified detector's predictions for its owned class, with explicit ID remapping. Do not concatenate both models' boxes for that class. Apply class-aware NMS and validation-calibrated thresholds. Bottle must remain absent from every output.

Accept a specialist only if evaluation of the complete combined pipeline on the same validation set meets the proposed improvement gate, preserves other classes, and reports added latency. A specialist that fails the gate is rejected. Evaluate the chosen unified or combined pipeline once on the reserved test cohort and report development-data limitations. No specialist is guaranteed to be deployed.

## Deliverables and verification

- An isolated V5 six-class preparation/training/evaluation CLI and Colab entry point based on current main; historical notebooks/results retained and clearly linked as seven-class history.
- A manifest and preparation receipt proving six-class names, no class-6 rows, unchanged source images, class support, exclusions and zero group overlap.
- A unified-trial ledger containing actual metrics and failures, three-seed confirmation, and a machine-readable specialist eligibility decision.
- Conditional specialist training plus ownership-based inference and whole-pipeline comparisons, implemented only after measured eligibility.
- Tests for mixed-image bottle removal, empty-label retention, invalid labels, overlap rejection, immutable receipts, eligibility refusal after missing/failed trials, class-ID ownership and bottle-free outputs.
- A real dataset preparation check, runtime smoke check and actual full training/evaluation runs before any improvement/completion claim. Smoke checks do not prove accuracy gains.

## Outstanding inputs and boundaries

The latest corrected complete dataset and exclusion ledger are not currently local. Their location was requested. The verified seven-class checkpoint is available, but its exact original split provenance is not.

MPS is the available default compute option; per-run timing must establish the expected duration. Do not purchase cloud compute or assume access to the user's interactive Colab runtime. Save progress so training can survive interruptions. Any switch to externally billed compute needs a concrete user choice.

No training or implementation has occurred under this design. Numeric targets, budget, and proposed run recipes await review. Results may show unified success, specialist success, or no reliable improvement; report the measured outcome without promising an accuracy increase.
