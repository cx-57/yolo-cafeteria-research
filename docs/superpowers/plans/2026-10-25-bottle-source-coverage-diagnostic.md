# Bottle source-coverage diagnostic before another training choice

User transition criterion: validated bottle AP50>=90% before moving on.
Continue tracking AP50–95, including the prior64–70% aspiration, and preserve
the balanced84.071068% /62.060566% checkpoint. Do not force data or overfit.
Full training remains paused. No other class is being advanced.

V34's fresh monitor-selected epoch7 scored80.084152% /63.732109% on the
36-photo Open Images monitor but82.432136% /60.044393% on the frozen benchmark.
V36's selected epoch8 scored80.895850% /63.940008% on that monitor and
81.917030% /59.389091% on the benchmark. Both were below the balanced model.
The benchmark has61 trash_detection bottle boxes and6 plate boxes; all12
positive monitor photos are Open Images. This supports a source-coverage
hypothesis, not proof of its cause or permission to change benchmark labels.

1. Preserve the original246-photo/67-box benchmark and permanent36-photo monitor.
   Keep V34/V36 declarations, source files and completed choices immutable.
2. Complete review and conservative overlap screening of V37's61 newly reviewed
   Open Images photos/104boxes and V38's47 fresh litter photos/60boxes. Source
   community annotations are not maintainer reviewed; all cut boxes, cups,
   grouping/missing-object and material uncertainties stay whole-photo holds.
3. Independently review fresh V39 and V41 litter backgrounds. V41 uses Commons
   per-photo license/artist metadata while Flickr requests remain stopped after
   HTTP429. A source category list
   without bottles is never absence proof. Require two whole-scene AI reviews,
   actual source-photo identity/geometry and attribution, plus overlap screening
   against every protected source and all V37/V38 candidates. Keep every flag.
4. Before any model measurement, use fixedseed20261025 on sorted eligible IDs to
   reserve16 fresh litter positive photos and16 verified-absence litter photos.
   Eligibility amendment made before any model measurement: every positive
   photo's source boxes must have area>=1024pixels after uniformly resizing its
   full image to max side640. Data-only scale inspection found original boxes
   0small/38medium/29large, versus fresh45photos/57boxes30small/25medium/2large.
   Twenty-three fresh photos/25boxes meet this rule. This avoids introducing
   smaller objects absent from the benchmark into the primary comparison;
   smaller photos remain possible future training data, not discarded or edited.
   The new cohort will still have a different medium/large mix, so this cannot
   isolate source effects causally. The rule and seed must freeze before inference.
   If fewer pass, stage further fixed source samples before measuring; do not
   lower the support requirement based on scores. Freeze all original source
   boxes, reviews, group IDs and dataset fingerprints. These32 photos and any
   related capture/byte/pixel aliases must never enter future training.
5. Compile a separate litter diagnostic dataset. Keep the existing36-photo
   monitor unchanged and report the two cohorts separately. The fresh litter
   cohort is development evidence, not an independent NHS test. Public-source
   pretraining exposure and incomplete capture detection remain limitations.
6. Freshly reload exactly the balanced model88877bbf..., V34 winner641d162e...
   and V36 winner4afd3108... on the new litter cohort at640,batch4,NMS0.5,noTTA.
   Bind each result to its exact checkpoint SHA and source/label fingerprint.
   Existing old-monitor receipts may be referenced with their hashes, clearly
   attributed to that original cohort. No original-benchmark inference here.
7. Assess the predeclared hypothesis: did the newer models' Open Images gains
   accompany worse litter performance? Do not assume it will be confirmed.
   Use the outcome to define a substantive next bounded training intervention
   and predeclare source-aware monitor gates. No threshold/hyperparameter sweep
   on the original benchmark, and no full-training launch.
8. Test permanent monitor/candidate exclusion, stale review/screen rejection,
   source annotation closure and saved-score attribution before diagnostic
   inference. Independently review the prepared dataset/evaluation code; record
   results on GitHub. A90% move-on claim still requires the unchanged original
   benchmark, not either development monitor or a mixture of checkpoint scores.
