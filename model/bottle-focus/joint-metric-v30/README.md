# Bottle joint-metric trial V30

Target AP50>=90% and AP50–95>=64% together; 70% localization is aspirational.
Keep the existing 246-photo/67-box development benchmark frozen. All reported
results remain development model selection, not independent NHS test scores.

Pin the V19 balanced specialist 88877bbf…1518 (historical ordinary 84.0711% /
62.0606%), while preserving the AP50 incumbent 2cd1efe2…2289 (84.0784% /
60.9019%). The initial V30 model is freshly checked before training.

Reuse V28 reviewed sources, with one conservative training-only hold for
uncertain object extent/attached packaging. Neither reviewer confirmed a
definite localization error. Original source images/labels are unchanged.
Final counts: 1,114 source photos / 470 source boxes; 187 derived crops from
102 source photos; 1,956 total exposures; 1,301 distinct training image views
and 731 view annotations. Reuse the tested focus_crop helper at 60% image
size around small targets; retain intersecting labels, omitting subpixel
intersections as the helper specifies. Replace two of eight repeated views
or one of four, retaining full-image views. The two crop-preview files are
QA sheets generated before the single-photo hold, including its uncertain
example. They are not an exhaustive human review.

Head-only AdamW 0.00001, fixed BatchNorm,640 pixels, batch 4/nbs 16, seed 43,
mosaic0.5 and scale0.5. This screens combined quality/crop training from
the balanced checkpoint; it is not an isolated causal crop comparison.

Retain separate AP50/localization/native-joint winners and evaluate latest
weights too. Fresh ordinary640/batch4/NMS0.5 checks use immutable hash-bound
snapshots. metrics-history.json retains outcomes and frontier.json retains
nondominated pairs. The legacy incumbent is promoted only if neither
metric falls. Both target thresholds must be met by one actual report.

continue_training.py runs three-epoch phases, up to 40 epochs. A 0.2 percentage-
point increase in either best metric resets operational patience; three
phases without that progress stops the screen. This is not statistical
significance or convergence. Plateau/schedule/storage limits are honest
below-target states, including across restarts; no automatic forced retry.
A 512 MiB free-space floor stops new phases.

Six focused tests passed after observed RED, covering joint thresholds,
tradeoffs, checkpoint choices, artifact integrity, plateaus and restart.

The freshly reloaded V30 baseline confirmed84.0711% AP50 and62.0606%
AP50–95 on the unchanged246/67 data. Live checks verified head-only
training and all81 BatchNorm statistics fixed. The first phase is active;
no trained-model improvement or target achievement is claimed.
