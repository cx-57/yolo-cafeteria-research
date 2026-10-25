# Reviewed neck/head monitor continuation

User goal remains90% fixed-benchmark AP50, with the prior64–70% AP50–95 goal,
without forcing results. Full training remains paused. This is a separate,
bounded neck/head continuation with the complete YOLO11s backbone0–10 frozen.

V34 completed82.432136% /60.044393% on the unchanged246-photo/67-box benchmark.
Its fresh monitor choice80.084152% /63.732109% did not transfer across domains.
The protected balanced84.071068% /62.060566% model stays unchanged. V36 does
not resume V34, extend its budget, alter frozen annotations or claim90%.

V35's independent AI scene/crop reviews plus source metadata and conservative
overlap screen retained36 new photos:16 Open Images positives,9 Open Images
backgrounds and11 TACO positives, with40 positive boxes. All29 overlap flags
remain held. The protected corpus covers1,448 distinct photos /1,905 role aliases.
Source admission joins the original AI-reviewed image SHA, complete official
annotations, original dimensions/orientation and per-photo attribution/license
metadata. TACO raw annotation SHA is checked, all category5 source boxes retained,
and unreviewed category4 mixed scope rejected. No AI review certifies material.

All V34 training exposure counts remain unchanged; each new photo receives4
scheduled exposures. Dataset603 photos /561 boxes /2,244 scheduled exposures.
Repeats are not independent photos. Permanent monitor36 photos /17 boxes and
the original benchmark remain unchanged, with monitor IDs/byte/pixel aliases
explicitly excluded from admission. Monitor selection remains development and
does not establish independent NHS accuracy or absence of overfitting.

Initialize balancedV19 SHA88877bbf...1518. Freeze11 includes backbone C2PSA;
all BN statistics fixed. Train neck/head only with AdamW3e-6,batch4/nbs16,
640,seed43,mosaic0,scale0.25,translate0.05,maximum9epochs. Keep native monitor
AP50/AP95 winners plus latest; fresh initial/winner/latest monitor checks use
640,batch4,NMS0.5,noTTA. Select maximum AP50+AP95, initial/earliest tie preference.

Another original-benchmark measurement is allowed only if the selected fresh
monitor pair meets BOTH AP50>=0.8058415166393892 and AP95>=0.6373210856519369.
This is at least+0.5 percentage-point AP50 over V34's monitor winner with no
localization regression. Otherwise archive without another benchmark inference;
unmeasured chosen-checkpoint benchmark metrics stay null and the old benchmark
reference is explicitly attributed to its own weight SHA. At most one distinct
candidate may be benchmarked. Global promotion requires both metrics nonregress;
90/64 must come from the same saved and freshly checked model.

237 project and16 focused checks passed locally, with no failures/skips.
Admission/gate controls received RED-to-GREEN checks. Actual finalizer controls
cover below-gate no-inference behavior, no cached-score misattribution and saved
report recovery. Actual CPU YOLO11 model checks enforce freeze11 and reject
freeze10. Layer11 is parameter-free Upsample; neck checks use parameter indices
11–22. Independent code/data review and follow-up identity-join review found no
concrete P1/P2. Frozen246 photo/label/manifest hashes and protected weight hashes
were rechecked unchanged. Runtime checkpoints stay outside iCloud.

Completed9epochs. Fresh monitor winner epoch8:
80.895850% AP50 /63.940008% AP50–95,
SHA4afd31080d871eea5993aade1fb871697cdc009e5b2f2f77c75e3b11d5c41e93.
This passed the previously declared monitor gate, so exactly one new candidate
used the fixed benchmark. Result81.917030% /59.389091%: both regressed against
the protected balanced model, which was retained.90/64 remains unmet. No full
training was started and no extra epochs or benchmark candidates were added.

Coverage diagnostic using source labels only (no further inference) found
61/67 benchmark bottle boxes in trash_detection and6/67 in plate, whereas all
12 positive monitor photos are Open Images. Both V34/V36 improved that monitor
without transferring to the fixed benchmark. This motivates reviewing broader
fresh litter scenes and monitor coverage before another training choice. It
does not justify changing benchmark labels/split or claiming independence.
