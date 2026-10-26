# Bottle logit interpolation implementation plan

> Execute inline using superpowers:executing-plans; user authorized continued autonomous optimization.

**Goal:** Test whether shrinking the V45 epoch-one classifier update retains cross-source gains without its 0.078 percentage-point litter AP95 regression.

**Architecture:** Interpolate only the six native terminal classifier tensors from balanced incumbent toward V45 epoch one, at fixed alpha 0.25 and 0.5. All other weights and buffers must remain bitwise identical through save/reload. No training, new data, or label changes. Reuse frozen V45 data verification and exact evaluation settings. This is another development experiment on repeatedly used monitors, not an independent test.

**Alternatives considered:** More epochs worsened the V45 localization tradeoff. Broad feature training regressed in V42/V44. A smaller logit update tests the observed score-ranking issue directly, cheaply, with two candidates only.

**Selection:** Same immutable V45 gates: retention metrics each within .005 of incumbent; all four older-source metrics nonregressing; mean older-source gain at least .01. Rank eligible candidates with existing V45 protocol, alpha .25 wins exact ties. At most one distinct original-benchmark candidate. Benchmark uses original 246 photos/67 boxes with all frozen image/label SHAs checked. Promotion requires both AP50 >= .8407106817291071 and AP95 >= .620605656065828; move-on requires same checkpoint AP50 >= .9. Do not edit gates after measurement.

## Tasks

- [ ] Add test for interpolation exact protected tensors, endpoint preservation, invalid alpha/state rejection; run failing, implement `blend.py`, rerun. Use real incumbent/candidate save/reload smoke.
- [ ] Implement `campaign.py`: freeze code, parent declaration, exact endpoint SHAs, scope/data/reference receipts and fixed alphas before evaluation. Reverify all locks on restart; cache scores only with exact model/cohort/settings/declaration match. No global changes to previous experiment code. `--prepare` creates declaration; normal execution validates it and runs bounded assessment.
- [ ] Independently review code, run focused checks, then execute. Verify hash/support/geometry receipts and unchanged protected benchmark. Archive V45 and V46 results and push authorized code/evidence; weights remain local.

**Review focus:** Wrong endpoint; silent nonlogit drift on serialization; stale saved scores; restarting with different declaration/candidate; bypassing failed source gates. Tests check interpolation failures, actual checkpoint roundtrip and protocol rejection. Existing V45 gate/lifecycle tests remain unchanged.
