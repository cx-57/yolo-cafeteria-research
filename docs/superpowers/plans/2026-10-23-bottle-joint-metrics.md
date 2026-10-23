# Bottle joint-metric implementation plan

**Goal:** Pursue bottle AP50 >= 0.90 and AP50–95 >= 0.64 on the unchanged 246-photo/67-box development set; 0.70 AP50–95 is aspirational. Do not manufacture scores or sacrifice one metric silently.

**Approach:** Preserve historical AP50-first runs. Let the active V29 phase finish, then replace its old 85% controller. Its first two saved phases scored 83.8332/ 60.5537 and 84.0521/ 60.4837 percent. The ordinary V19 checkpoint offers a better localization starting point: 84.0711/62.0606, versus the AP50 incumbent's 84.0784/ 60.9019. Preserve both.

The next isolated V30 trial uses V19 initialization, V28 reviewed data, and bounded training-only focus crops. Reuse scripts/experiment.py:focus_crop. Replace two of eight repeated context views, and one of four repeated new-data views, when a source has a target below 4% of photo area. After initial crop QA, conservatively hold one source with uncertain attached packaging extent (not a confirmed label error). Keep all other 1,114 source photos; final training has 1,956 exposures and 187 crop views from 102 sources. All eligible photos have EXIF orientation 1; crops retain intersecting annotations using the existing helper, omitting subpixel intersections per that helper. These are augmented views, not new photos. No validation crop or relabeling.

Head-only AdamW 0.00001, fixed BatchNorm, batch4/nbs16, 640 pixels, seed43 and mosaic0.5 retain the established memory-efficient setup. This screens the combined training intervention from a fresh matched initialization baseline; it is not a causal claim about crops alone. Full-model and longer-source-data options remain possible if this screen is unpromising.

## Tasks

- [x] Add meaningful joint-target and Pareto-retention tests; observe RED then GREEN.
- [x] Build audited crop views with geometry and source-hash verification. Reuse existing crop tests and inspect real previews.
- [x] Track separate native AP50 and localization checkpoints; freshly evaluate both at ordinary640/batch4/NMS0.5. Retain immutable weights and all nondominated metric pairs. Promote the legacy incumbent only if neither metric regresses.
- [x] Add bounded phase continuation with exact joint success, restart checks and honest plateau/schedule-end states. A 0.2 percentage-point gain in either best metric resets operational patience; three phases without such progress stops this screening run. This is not a statistical significance rule or a convergence claim.
- [ ] Run project and focused tests, independent review, finish/archive V29, launch V30, commit/push actual code, audits and outcomes.

## Constraints

Validation membership, image bytes, 67 labels, source fingerprint and protocol remain frozen. Source review is AI assisted, not human verification. No independent NHS test claim. Targets must be met by the same saved checkpoint and inference configuration. Preserve failed trials; do not extend schedules blindly or round a below-target score upward.

## Verification ledger

- Independent review P2: restart bypassed a plateau stop. Actual failing lifecycle test observed; terminal-state guard added, interrupted completed epochs reconciled once, six focused tests pass.
- QA hold: uncertain object extent/attached packaging in trash_detection__train__000032_JPG.rf.cbf03a9b729557ed636289ae8c39e1e3.jpg; neither reviewer confirmed a definite incomplete box. Hold is training-only.
- Final audit distinguishes 470 source boxes from 731 generated-view annotations; 1,301 distinct views are not 1,301 new source photos.

- Fresh V30 baseline confirmed84.0711/62.0606. All246 image/label hashes still match the frozen receipt. Head-only training is active; target not achieved.
- Full235 project tests and six focused tests pass. Code review and image QA completed.
