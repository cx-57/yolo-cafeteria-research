# Bottle hard-negative logit calibration plan

> Execute inline with superpowers:executing-plans; use independent code/data review. Optimization authorized; max20epochs per user, this trial3epochs.

**Goal:** Test whether a much smaller, hard-negative-focused classifier update can avoid V44's cross-source localization regression.

**Evidence:** V44 froze spatial weights but trained all classifier features; OI83.13/67.30 and litter49.31/26.38 versus baseline61.80/51.38 and47.56/28.20. LitterAP95regressed, so gate retained initial and skipped original benchmark. Training-only mining atconfidence.1 found27of142empty-label photos with false proposals; high-scoring cases include papercups,cigarettes,feather,bags,vape. Material/scope ambiguity remains for actual bottle-like negatives; quarantine these instead of reinforcing them.

**Architecture:** Start again from balanced88877bbf…, preserve V44's24retention photos and68older monitors, and reuse verified immutable V44training ancestry. Keep every nonlogit tensor fixed, including spatial and classifier features, all BatchNorm buffers and EMA fixed state. Train only three native1x1classifier output convolutions (six parameter tensors). Userexplicitlyconfirmed ONLYdisposableplasticdrinkbottles. Whole-photo/whole-group quarantine122legacypositiveparentsnotacceptedbybothscope/completenessreviews, plusambiguoushardnegatives7,12,14. Clear glasswine andnonbeverageketchup arevalidnegativesunderthisscope; onlyuncertaincasesheld. Add eight scheduled exposures per accepted clear nonbottle hard negative; no new parent images or label edits. Root/second review bind actual photo SHA and mining order.

**Recipe:** Max3epochs,b4/nbs16,640,MPS,AdamW1e-4,BNfixed,seed43,mosaic0,scale.25,translate.05,cosineLR,warmup1,patience0. Stop afterepoch2if either retention AP metric drops>.03; terminal restart recovery from V44 retained. User20epoch cap respected.

**Selection:** Same source-separated floors as frozen V44: retention each≥initial-.005, all4older-source AP metrics nonregressing, mean source gain≥.01. At most3distinct native candidates, at most1gated original benchmark candidate. No retroactive relaxation or threshold/TTA sweeps. Promotion requires original AP50/AP95 ≥84.071068%/62.060566%, move-on requires AP50≥90. Public source monitors and previously-seen retention are development diagnostics, not independent NHS tests.

## Tasks / interfaces / review focus

- [ ] Bind mining/source/review identities and quarantine scope-ambiguous negative groups; `data.verify()->audit` proves only exposures changed and all92monitor/source aliases remain excluded. Test unauthorized reweight of positive/held/unreviewed/stale-source records fails.
- [ ] `geometry.allowed_names(model)` identifies only actual terminal Conv2d(1x1,nc=1) by module identity. Test exactly6mutable tensors, changed scores, unchanged boxes and intermediate classifier state after real optimizer/EMA/save steps. Unsupported heads fail closed. Actual source-list batching test prevents the observed142-image allocation despite requestedbatch4; mine with explicit4-image chunks and MPSmemory cap.
- [ ] Reuse V44lifecycle/metric tests and assert actual final projection trainability on real training path; freeze code/recipe/data/reviews before optimizer steps. Independently review and run focused/project checks.
- [ ] Run3epochs, source-separated assessment, preserve exact results and incumbent; never label source-only gains as benchmark improvement.
