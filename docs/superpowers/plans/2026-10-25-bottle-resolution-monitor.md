# Bottle resolution and fresh monitor implementation plan

> Execute inline using superpowers:executing-plans, under ongoing autonomous optimization authorization.

**Goal:** Test one896pixel ordinary-inference setting on the current incumbent, without further weight tuning or repeated original-benchmark selection.

**Evidence:** V46score adaptation andV48box-only adaptation didnotimprove the incumbent. Larger-input experiments recorded earlier used older checkpoints and/or augmented inference; this is a single ordinary-inference test on balanced88877 with matched640 controls. Small-context bottles may benefit from more input detail; a gain is a hypothesis, not a promise. NewTACO source QA rejected most images for metadata, material, coverage or extent issues. Keep original source labels unchanged; do not force questionable data into training.

**Architecture:** Reserve every V49twice-reviewed source photo that passes geometry and conservative overlap/sourcegroup screening as a permanent new development monitor. Require at least4photos/4boxes before running. Never train on these photos/groups/aliases. Preserve the existing92monitorphotos and original246/67benchmark. These public development sources do not establish independentNHS performance or exclude pretraining overlap.

**Recipe:** Weights88877 fixed; onecandidate896ordinary, matchedcontrol640, batch4,MPS,NMS.5,noTTA,workers0,Ultralytics8.4.175. No optimizer, model blending, source relabeling or sweep. Record input size explicitly;896is a new inference protocol, not a640score. Measure unchanged-model hashes and latency.

**Gates:** Use four cohorts:24previouslyseenretention,36OI,32litter,andfreshV49positive-onlymonitor. Retentioneach>=matched640-.005; all6APmetrics on the other3cohorts nonregressing,mean gain>=.01. Fixedcandidate only, max1original896evaluation ifgatespass. Main labels/images unchanged; compare against84.071068%/62.060566%ordinary640. Promotion/recommendation requires both nonregression;90%AP50move-onmuststate896ordinaryinference on samefrozen246photos andcheckpoint. No rewriting historical640scores. Control original score reused, not rerun.

## Tasks

- [ ] Bind V49reviews/adjudication/geometry/overlap/source metadata; generate only new monitor snapshots and exact normalized full source bbox rows. Screen group/byte/pixel/near aliases against1638historical plus34V48newtrainingphotos. Freezepermanentmonitor manifest and allimage/labelSHAs before inference.
- [ ] Add failing gate/receipt/attemptlock tests; implement single-candidate protocol and campaign. Freeze code, parent ancestry, data, recipe and original frozen hashes. Reject stale caches/config changed on restart; original attempt lock binds weights ANDresolution/settings, not weights alone.
- [ ] Run focused checks and independent review; evaluate matched controls/candidate only, then gate. Verify alloldweights/labels unchanged, archive and push exact metrics. No training launch and no claim of90withoutactualmainresult.
