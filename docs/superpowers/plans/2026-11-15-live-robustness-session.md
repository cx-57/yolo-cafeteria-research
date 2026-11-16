# One-hour live robustness session — 2026-11-15

Requested start: approximately18:02 America/New_York, target approximately19:02.
User focus: utensils fluctuate to wrapper; bread is missed. Additional explicit
constraint: do not overfit. No original data, annotations, splits, weights or
frozen research results may be modified. No new training without explicit
permission. Servo operation remains disabled.

Predeclared scope: investigate preprocessing/latency and UI stability, compare
frozen existing detectors rather than training new ones, and add reproducible
camera diagnostics. Do not tune on every live frame or expand the trial set
until a desired score appears. Candidate family limit: existing unified,
frozen four specialists, existing six-class correction composition plus frozen
bottle (only if class/latency checks permit). Keep earlier comparisons as
exposed development evidence, not a new independent evaluation.

Camera session A: diagnostic observations of bread/utensil/wrapper, with user
confirming actual object. Frame bursts and views of the same physical item
belong to one group; never split adjacent frames across train/evaluation.
No training is part of this session. Freeze decisions before session B.
Camera session B: different physical items/view groups, evaluation-only;
record errors as well as successes. If a failure is used to change the system,
it becomes development evidence and a fresh evaluation group is needed.

- [x] Inspect actual camera failures and prepare local evidence capture.
- [x] Audit food/utensil coverage and inference paths without modifying research.
- [x] Measure strongest already-trained candidate(s), preserving seven classes.
- [x] Fix independently justified software defects with regression tests.
- [x] Freeze selected settings before fresh live evaluation; report limitations.
- [x] Run full tests, verify checkpoint hashes and document exact changes,
      elapsed work, failed checks and remaining human observations.

Ruling: user requested stop/reverse liveexperiments at18:54; camera was stopped and all further work remainedoffline. No fieldaccuracy claim is possible for the exactbreadcrumb/orange cases. Rejected fullframe tiling; fixedROIzoom stays optin/unconfigured. Largecandidate fullcomparisonwas stopped, not promoted. Defaultthreshold andallweightsunchanged. See docs/live-robustness-session.md for evidence andlimitations.
