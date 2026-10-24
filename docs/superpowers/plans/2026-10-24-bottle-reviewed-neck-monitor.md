# Reviewed neck continuation, with benchmark reuse gate

V34's predeclared monitor choice (epoch7, SHA641d162e...) freshly measured
82.432136% AP50 /60.044393% AP50–95 on the unchanged246-photo/67-box benchmark.
It is below the balanced84.071068% /62.060566% checkpoint, which stays protected.
Fresh monitor improvement alone did not transfer across these source domains.
Do not extend V34 or alter its declaration, data, labels or completed choice.

Continue the user's bottle goal through a bounded experiment; full training
remains paused. No promise of90% or absence of overfitting is made.

1. Finish V35 source screening. Keep every overlap flag and metadata/geometry/
   appearance hold. Bind two AI reviews, official complete source annotations,
   per-photo provenance, source geometry and protected corpus identities before
   admission. Original benchmark and permanent36-photo monitor never train.
2. Build a separate training-only expansion from V34's exact exposure list,
   retaining all prior exposure counts. Add passing positive and negative photos
   four times each. These repeats are not independent photos. Keep the same
   monitor and frozen benchmark. Do not modify V34's training list or weights.
3. Verify RED-to-GREEN admission controls for source-box tampering, protected
   monitor aliases, stale screen receipts and partial whole-photo annotation.
4. Start from protected balancedV19 weights. Freeze full backbone layers0–10
   (freeze11, including C2PSA); train neck/head only, all BN statistics fixed.
   AdamW3e-6, batch4/nbs16,640,seed43,mosaic0,scale0.25,translate0.05,
   fixedmaximum9epochs. This is a bounded continuation, not full backbone training.
   Prior neckV31 used noisier backgrounds and mosaic0.5 at1e-5; this is a
   substantive data/recipe change, not an isolated causal ablation.
5. Retain native monitor AP50/AP95 winners and latest; fresh monitor checks at
   ordinary640,batch4,NMS0.5,noTTA. Select maximum AP50+AP95, with initial/earliest
   tie preference. No checkpoint or epoch may be chosen using benchmark output.
6. Before a new benchmark measurement, require fresh monitor AP50 to exceed the
   V34 monitor winner by at least0.5 percentage point, and AP95 not regress:
   AP50>=0.8058415166393892 and AP95>=0.6373210856519369. Otherwise archive the
   trial without another benchmark measurement. This gate is declared in advance;
   repeated monitor selection is still development, not an independent test.
7. At most one distinct selected candidate may use the original benchmark;
   persist the choice/declaration across restarts. Promote only if both fixed
   benchmark metrics do not regress.90/64 must belong to one saved checkpoint.
8. Independently review admission/controller changes; run relevant controls,
   verify all frozen photo/label/protected checkpoint hashes, record source and
   results on GitHub, and execute the bounded experiment. No full training launch.
