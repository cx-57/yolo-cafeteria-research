# Bottle specialist experiments — 2026-10-23

Latest verified result (2026-10-25local;2026-10-26UTC): **two-model ensemble85.3558%AP50/62.1329%AP50–95**, same246photos/67boxes. Exactrecipeinbest-ensemble/recipe.json. Thisisnotasingle-checkpointscore;single-modelbalanced84.0711/62.0606remainsunchanged. AllfourdevelopmentcohortsourcegatespassedbeforeoneV51mainprobe. Latestuser20minaspiration86–87/~65notfullymet. V52devicebugfixedinV53,whichfailedsourcegatesandskippedmain;ensembleretained. Nooptimizertrainingduringthis20minwindow.


Current user target: **90% bottle AP50 and at least 64% AP50–95**, aiming toward 70% localization where supported. Both must come from the same saved checkpoint and inference settings. Do not force the target. Best AP50-first incumbent remains **84.0784% / 60.9019%** at ordinary640, CPU batch4, NMS0.5; a retained balanced checkpoint has **84.0711% / 62.0606%** at ordinary640, MPS batch4, NMS0.5. Neither meets the joint target.

The fixed development validation set has **246 images and 67 bottle boxes**.
All validation images, bottle annotations and splits are preserved.
Source fingerprint:
`6668d2b7f0b16243640c14efdbb6bdffbb7f83fc880459befd58e8afcbfa2296`.
Validation protocol:
`29423591896198d41f5bd16392b51f06c7c3d7254f8a903b3dda74569cfe7de7`.
These are development results, with no independent test claim.

| Saved model / experiment | Ordinary AP50 | Augmented AP50 |
| --- | ---: | ---: |
| Pretrained YOLO11x, bottle output only | 40.21% | 45.65% |
| YOLO11s head fine-tune, best of eight epochs | 67.11% | 69.08% |
| YOLO11x head fine-tune, two completed epochs | 60.28% | 60.23% |
| YOLO11s context resampling, best of eight epochs | 68.39% | 71.33% |
| YOLO11s context resampling, best of 23 epochs | 73.67% | 71.22% |
| YOLO11s gentle feature adaptation, best of eight epochs | 82.08% | 82.14% |
| Same checkpoint, NMS IoU 0.5 | — | 82.65% |
| YOLO11s head-only 35-photo supplement, best of six epochs | 82.99% | 80.75% |
| Half/half head average, TTA NMS IoU 0.5 | 83.02% | 83.35% |
| Same head average, ordinary NMS IoU 0.5 | 83.68% | — |
| Head interpolation alpha 0.75, ordinary NMS IoU 0.5 | 83.73% | — |
| Quarter quality-head blend, ordinary NMS IoU 0.5 | 84.07% | 80.68% |
| Context-balanced head, best saved model of 12 epochs, NMS IoU 0.5 | 83.14% | — |
| Half context/quality head average, CPU ordinary NMS IoU 0.5 | **84.08%** | — |

All scores are independently reloaded evaluations. The non-averaged
head-only checkpoint’s training-time validation score was 83.05%; the saved-model score above takes precedence.
Training validation uses a different batch size and saved checkpoints are
serialized at half precision, so the two scores need not match exactly.

The verified incumbent is retained locally at
`verified-incumbent/bottle.pt`; its settings and SHA-256 are recorded in
[`verified-incumbent/verification.json`](verified-incumbent/verification.json).
SHA-256: `2cd1efe2d7c187df05ed223ba2a18e147393358179f223c264663a2114812289`.
Existing Git ignores exclude model weight files; reports and experiment code
are versioned, while checkpoints remain local.

Completed comparisons:

- Unfreezing later features at learning rate 0.0001 did not beat the
  initialization in three epochs.
- Inference at 960 and 1280 pixels did not beat 640-pixel augmented inference.
- A training-only BatchNorm recalibration pass changed no learned parameters,
  but reduced validation AP50 for both tested checkpoints.
- Training-only resampling repeats 124 existing bottle-positive context
  photos four times per epoch. Every other training photo remains included
  once. There are 1,056 unique training images, 1,428 exposures and 890 bottle
  exposures per epoch; no new photos were admitted.

Context resampling completed 23 epochs. Its epoch-16 averaged candidate
reached 72.73% augmented AP50. Gentle feature adaptation then started from
the verified 73.67% ordinary-inference checkpoint, retaining fixed BatchNorm
statistics and reducing the learning rate to 0.00001. It completed 13 epochs; epochs 9–13 did not improve on epoch 7.
At 832 pixels, ordinary/augmented AP50 were 74.72%/80.16%; 640 pixels
with tighter NMS reached 82.65%.

The new `external-data-v7` experiment adds 19 freshly downloaded LVIS
photos with 36 plastic beverage bottle boxes to training only. The
AI-assisted visual reviews are explicitly not human verification.
Validation remains fixed at 246 photos / 67 boxes. A separate 548-photo
Mendeley source was sampled and rejected as supplied after 24 photos
showed repeated scenes and incomplete/out-of-bounds annotations.
The 19-photo supplement trial completed six epochs without beating its
starting ordinary AP50; the original checkpoint was retained. A combined
35-photo / 65-box supplement is now being tested with the feature layers
frozen and only the detection head adapting at learning rate 0.00003.
The combined supplement trial completed six epochs. Its epoch-3 checkpoint
reached 82.99% in independently reloaded ordinary inference. Default
multi-scale TTA reached 80.75% (NMS 0.7) / 80.63% (NMS 0.5), and a
full-resolution horizontal-flip variant reached 81.00%; these did not improve
on ordinary inference. A single half/half head average reached 83.02% ordinary / 83.35% augmented
AP50. Corresponding frozen backbone weights and all BatchNorm running
statistics were identical before averaging; only head parameters differed.
The head-only run completed epochs 7–8 without beating epoch 3. The
uncompleted epoch 9 was interrupted for a training-label audit; optimizer
state through epoch 8 is preserved. All 754 bottle-negative training photos
are being screened for possible missing targets before further fine-tuning.

The latest `hard-negative-polish-v11` trial uses 41 external photos with
80 bottle boxes, five reviewed training-photo holds and 12 repeated hard
negatives. A training-only scan found one definite missing-bottle-label photo
and four uncertain photos; no original data or validation labels were changed.
Mosaic is disabled and scale jitter is 0.25. Six epochs completed without
improving the ordinary control; the best newly trained native score was
81.02%. The original saved checkpoint is retained. The
`gentle-feature-polish-v12` trial unfroze layers 10 onward at learning
rate 0.000003 on the exact same training copy. Its native AP50 declined
80.22% / 76.92% / 76.12% across three completed epochs, so epoch four was
interrupted; optimizer state through epoch three and the control are retained.

The `expanded-context-v13` trial adds 16 further reviewed LVIS photos with
29 boxes (57 external photos / 109 boxes total), retains the five quality
holds, and removes extra hard-negative repeats. It uses 1,108 unique photos /
1,651 exposures and the previously successful head settings: freeze 23,
AdamW 0.00003, mosaic 0.5, scale 0.5, fixed BatchNorm running statistics.
Six epochs completed, best native 81.46%; independently reloaded augmented
AP50 was 78.35%. It did not improve the incumbent. V11 and V12 newly
trained candidates also scored 82.29% and 80.81% with augmented inference
and NMS 0.5. All validation bytes and labels remain unchanged.

The full 124-photo original context-positive cohort was reviewed with
targeted crops. Ten photos / fourteen boxes were conservatively held for
uncertain material/type; these are not ten confirmed label errors. The
`scope-clean-feature-v14` trial uses the five prior negative holds plus those
ten context holds, the same 57 external photos / 109 boxes, 1,098 unique
photos / 1,611 exposures, and 451 unique training bottle boxes. It starts
from the pinned pre-soup V6 checkpoint (edab…); the 83.68% global incumbent
remains protected. Later features adapt at 0.00001 with mosaic 0.5, scale
0.5 and fixed BatchNorm statistics. Native AP50 fell 76.67% / 71.36%
in two completed epochs, so epoch three was interrupted and the control
retained.

A fixed four-view rotation TTA test (0/90/180/270 degrees, full resolution,
NMS 0.5) reached 76.81% and was rejected. Both independent geometry/model
integrity tests and all 235 project tests passed; 246 adapter calls were
verified. Two neighboring head interpolation points, alpha 0.25 and 0.75,
are being evaluated as ordinary single saved models under the same settings
as the incumbent alpha 0.5. Alpha 0.25 scored 82.98%; alpha 0.75 scored
83.73%. Four-view rotation Weighted Boxes Fusion scored 83.12%;
two-view original/horizontal-flip fusion scored 80.89%. Both were rejected;
geometry/fusion tests and the full 235-test project suite passed.

The fixed quality-head blend uses 0.75 of the alpha-0.75 V8 head average
and 0.25 of the V11 quality-trained head. Frozen backbone tensors and all
BatchNorm statistics were identical. Freshly reloaded ordinary inference
reached 84.0711%; default augmented inference reached 80.6760%. The exact
ordinary checkpoint is retained. This winner combines V8 and V11 lineage;
the V11 branch used 41 external photos / 80 boxes and five negative-image
holds. Later 57-photo and 15-hold copies are separate trials.

The active `context-balance-v20` trial retains the scope-screened 1,098
unique-photo / 451-box pool, repeats 114 original context photos eight
times, and includes every other photo once. Its 57 external photos each
appear once; there are 1,896 exposures. It uses seed 43, pinned edab
initialization and the frozen-feature head recipe at 0.00003. The verified
84.07% global incumbent is protected. A half/half quality-head blend is
checked on CPU with its device explicit; it scored 83.45% and was rejected.
The context-balanced run completed 12 epochs; epoch 3 remained best. Its saved model reached 83.1407% ordinary AP50 at NMS IoU 0.5 and 82.6453% augmented AP50 at NMS IoU 0.7. Phase-12 checkpoints and reports are preserved.

Regression-only and classification-only branch blends scored 83.7310% and 83.8528%; both were rejected. A half parameter average of the 84.0711% quality model and the context-balanced epoch-3 head reached **84.0784%** on a fresh CPU evaluation. The frozen backbone and BatchNorm statistics were identical. This normal single-model checkpoint is the current incumbent. Default NMS IoU 0.7 reduced its AP50 to 83.3168%.

The larger `medium-head-v24` trial has now started from official pretrained YOLO11m weights with 20,053,779 parameters after single-class projection. A real non-square input confirmed unchanged decoded boxes and the original COCO bottle scores. It uses the exact V20 reviewed data and fixed validation, head-only training, fixed BatchNorm statistics, learning rate 0.0001, 640 pixels and batch 2. The six-epoch head phase finished; epoch 3 remained best. Fresh ordinary/augmented checks reached 68.8293%/72.7199%. The 84.0784% smaller-model incumbent is protected.

Run using the existing local runtime:

```sh
PYTHONPATH=/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/py312:$PWD/scripts BOTTLE_PHASE_EPOCHS=6 .venv/bin/python -u model/bottle-focus/context-balance-v20/run.py
```

The cached dataset is prepared by
`validation-now/prepare_data_snapshot.py`. Individual experiment directories
contain their scripts, configurations, metrics, hashes and evaluation output.

## Paper-informed continuation

The original higher-scoring papers and their training/data differences are
recorded in [PAPER_REVIEW_20261023.md](PAPER_REVIEW_20261023.md). Numerically
higher external scores are not head-to-head results on our fixed set.
The small incumbent scored 83.6731% with ordinary NMS 0.6 on MPS, so that
setting was rejected. A matched ordinary MPS batch-one, NMS 0.5 control
scored 83.4980%. Official SAHI 0.12.8 sliced inference at original resolution
(384-pixel slices, 25% overlap, full-image prediction, NMS 0.5) scored
76.3749% and was rejected. All 246 source images and 67 boxes were retained;
740 slices and 246 additional full-image predictions were traced. Two
independent coordinate tests, including the actual YOLO loader, passed.

The V25 medium-model feature adaptation has started from the immutable
V24 phase-six best snapshot, SHA e01fc63b…8934. It unfreezes from layer 10,
uses AdamW 0.00001, fixed BatchNorm statistics, training batch 1 with nbs 16,
640 pixels, and up to 100 total epochs with mosaic disabled in the last ten.
Six-epoch review phases preserve optimizer state. Results remain pending;
this is a paper-informed experiment, not an exact paper replication.

A fourth LVIS staging/review batch yielded 40 usable photos. Initial AI
review marked 20 eligible; second review held three more for ambiguous
identity/partial targets or incomplete box coverage. **17 photos / 20 boxes**
remain eligible candidates. None has entered V24 or V25 training. The
validation-overlap heuristic flagged another already-held candidate. All
reviews are AI assisted and are not human verification.

The V25 adaptation was preserved after two completed epochs (native AP50
66.2860% and 68.0283%); neither beat its initialization. Its selected saved
checkpoint remains the 68.8293%/72.7199% ordinary/augmented baseline. This
was a short screen, not a claim that the planned 100-epoch recipe converged.

Inspection of the actual WaDaBa author notebook found a distinct fully
trainable recipe: effective AdamW 0.000909, 20 epochs and nbs 64; requested
0.01 was ignored by auto. Its output logs 3,200/800 images, not an observed
11,000-image YOLO run. The new `paper-recipe-s-v27` control uses those
optimizer/schedule settings on our unchanged data, with YOLO11s/batch 4
for this Mac. It starts from fresh projected COCO b28f0550, whose saved
baseline AP50 is 26.9615%. Live checks confirm all 9,428,163 feature/head
parameters trainable (only DFL fixed) and all 81 BatchNorm modules training.
Training is active; no result is claimed for its pending epochs.

The V27 control's first two native validation scores are 28.7408% and
29.8204%; these are not freshly reloaded scores and do not replace the
84.0784% saved incumbent. A rounded-prediction replay of small/medium
2:1 weighted fusion scored 79.8197%, versus its replayed small-model
control of 83.4997% (fresh control 83.4980%). This replay is exploratory
only, not live two-model inference or an independently validated ensemble;
it did not justify diverting the active training run.

A separate read-only validation review found one visibly unannotated clear
water bottle in a photo with an empty label and one strong visual glass-versus-
plastic scope mismatch among the labeled positives. Both received a second
AI review; material identity is not physically verified. Other ambiguous
containers remain uncertain. All 246 image hashes and label rows still match
the frozen manifest and protected weights are unchanged. See validation-review
for evidence, source hashes and review limits. Historical 246/67 scores are
retained; no corrected score or 85% achievement is claimed.

V27 has completed four epochs; native AP50 increased to 46.1055%. The complete
20-epoch run remains active, with fresh saved-model validation scheduled by
the runner. Commit 4a099127 passed both GitHub Pipeline tests jobs.

V27 has now completed all 20 planned epochs. Epoch 18 was best natively
(64.3836% AP50); freshly reloaded selected weights reached 64.5471% ordinary
and 62.7979% augmented AP50, both at NMS0.7. Phase-20 snapshots are preserved.
The hardware-adapted paper recipe did not beat the 84.0784% specialist;
this does not isolate the effect of learning rate from initialization and
normalization differences.

V28 admits the 17 round-four reviewed photos / 20 boxes to a separate
training copy: 1,115 unique photos / 471 boxes / 1,964 exposures. Source
image bytes, annotations, reviews and overlap screens are checked before
admission and on reuse. Two admission tests passed after observed RED;
all 235 project tests passed without skips. The planned head-only trial
starts from immutable V23 alpha0.5 weights with low-rate AdamW; original
validation and incumbent weights remain unchanged.

V28 completed its six-epoch head-only phase. Best native trained epoch 1
was 82.3511%; fresh ordinary640/batch4/NMS0.5 verification of the saved
trained checkpoint scored 83.4063%, SHA 7b26bf57…33b6e. Initialization
remains selected, and the 84.0784% incumbent is unchanged. Exact saved
backbone and BatchNorm statistics were checked unchanged. Phase 6
snapshots and metrics are archived.

V29 starts from the same immutable incumbent and exact V28 data, unfreezes
all features at AdamW 0.000001, and keeps BatchNorm statistics fixed.
Its sustained controller runs three-epoch phases through at most 40 epochs,
automatically checks saved ordinary640/batch4/NMS0.5 weights, archives
results, and stops only at an exact 85% crossing or an honestly reported
below-target schedule end. Review found a restart-after-success bug;
tests observed failure, then passed after adding preflight verification.
Incomplete interrupted phases require a matching fresh report before
more training. All 235 project tests and four continuation tests pass.
The controller is launched; V29 results are pending.

## Joint-metric continuation

V29 completed nine epochs and three saved-model checks. Its best retained
pair was 84.0521% / 60.4837%; no localization improvement. The old 85%
controller was superseded after its phase completed, with optimizer and
weights preserved. This is not a convergence claim.

V30 uses the balanced V19 checkpoint and training-only focus crops.
Independent visual review left one source photo's attached packaging extent
uncertain; it and its derived views are held, without editing any source or
validation label. Final copy: 1,114 original photos / 470 source boxes,
187 crop views from 102 sources, 1,956 exposures,1,301 distinct image views
and 731 view annotations. Crop QA previews were generated before that hold.

Separate AP50, localization and native joint winners are saved; the latest
checkpoint is also checked. Fresh immutable weights are evaluated at
ordinary640/batch4/NMS0.5. Nondominated metric pairs are retained, and the
legacy incumbent is replaced only when neither metric declines. Exact
90%/64% joint success, plateau, schedule and storage stops are distinct.
A reviewed restart bug that bypassed plateau was reproduced and fixed.
Six focused joint-policy tests pass; full project verification is recorded
in the experiment directory. No V30 accuracy improvement is yet claimed.

V30 completed three epochs before iCloud offloaded its 165-byte localization
selection ledger. Checkpoints and optimizer were intact. A bounded small-JSON
materialization fix passed 237 project tests and six joint-policy tests,
with independent review. Recovery validates saved candidates before further
training; this operational fix is not an accuracy improvement.

Recovery fresh phase3 checks: selected epoch1 =83.8031% /61.4286%;
latest epoch3 =82.5968% /60.2242%. No gain over84.0711% /62.0606%.
The controller reconciled phase3 with stale count1 and resumed its next
three-epoch phase; the exact joint target remains unmet.

## Continued bottle work, README workflow set aside

V30 completed6epochs with no gain. Fresh selected83.8031%/61.4286%,
latest81.7635%/59.0742%; optimizer retained and phase archived.
V31 prepares25 independently AI-reviewed photos/34 source boxes, giving
1139full-image training photos/504boxes/2056exposures. BalancedV19 start;
freeze11 preserves full backbone0–10 while neck/head train at1e-5, BNfixed.
237 project tests and11focused checks pass. No new accuracy gain claimed.

## Full training paused at user request (2026-10-24)

V31 completed3epochs and fresh saved checks regressed: selected82.3376%/
59.7363%,latest78.0905%/54.9265%. Preserved optimizer and archived trial.
Protected balanced baseline remains84.0711%/62.0606%; target90%/64% unmet.
V33 prepared579photos/538boxes/2148scheduled exposures, with166 newly
AI-reviewed background photos and23new Open Images photos/34source boxes.
237project/11focused checks pass; frozen246validation images/labels and
protected checkpoints unchanged. V33/full training has not started.

## Bounded monitor campaign resumed (2026-10-24)

User resumed bottle experiments toward90% AP50 without overfitting; full
training remains paused. V34 trains567photos/521boxes/2100exposures and
uses a separate36photo/17box continuation monitor for epoch selection.
Fixed9epoch ceiling, head-only/BNfixed. Predeclared monitor criterion, then
one distinct final246/67 benchmark candidate. Source aliases/capturegroups
removed, current+historical overlap screens reviewed.237project/10focused
checks pass. A reviewed interrupted-report recovery P2 was fixed. Native
monitor scores are never benchmark success. Trial launched; no gain yet.

## Disposable drink-bottle scope and continued optimization (2026-10-25)

User clarified onlydisposableplasticdrinkbottles; scope documented in docs/BOTTLE_TARGET_SCOPE.md. V42fullboundedrun regressed79.3971/57.9725 on original246/67benchmark. V44classifier-feature andV45terminalscore trials failed unchanged sourcegates and skipped original evaluation. V46halfV45epoch1logit interpolation passed sourcegates but main83.3384/61.0726 regressed; rejected. Protected balanced incumbent stays84.0711/62.0606;90%notreached. Source/retention high scores are not main benchmark scores.

V47staged80previouslyuninspectedofficialOpenImagesphotos with complete source annotations,positivePlastic/Waterimage labels,metadata andfixedseed1143. Root/independentAIwhole-scene reviews jointly accepted35;conservativeoverlapscreen against1638protectedphotos held1,leaving34parents/51boxes. V48adds onlythesephotos to unchangedV45schedule:567photos/595boxes/2027exposures. All92developmentmonitorphotos andoriginal246benchmarkphotos preserved. Nooldlabelrepairs.

V48launched bounded3epochAdamW1e-5terminalcv2boxprojection training frombalancedincumbent;6tensors/12480parameters withclassifier,features,BNandEMAprotectedstatefixed. Actualoptimizer/Trainer/savedcheckpoint checks and11focusedtests pass;independentreviewclean. Same retention/sourcegates,earlystop,max1originalcandidate,andjointpromotionfloors apply. NoV48accuracygainyet;full80epochmulticlasstrainingnotstarted. Publicmonitors are repeatedlyuseddevelopmentdata,not independentNHStesting.

V48completed3epochs: freshfirstcheckpointmatchedallmonitorbaselines;latestlitter47.5631/26.7782regressedAP95. Sourcegatesfailed,originalbenchmarkskipped,protectedincumbent84.0711/62.0606unchanged. Actualsavedweightschangedandprotectedstate+246benchmarkfilesverifiedposttraining.

V50reserves8newpreviouslyuntrainedlitterphotos/9boxes (V49indices2,8,9,17,22,29,40,51) afterdualAIQA,finiteboxchecks andoverlapscreenagainst1672protectedphotos. Newmonitorfingerprint ba927633e1c35e7e044eb9d8517c9f96b70c8a9bd34287e6347e5779115b7268. See permanent-monitor-registry.json: all100developmentmonitorphotos/groups/aliases remainexcludedfromfuturetraining. Single896ordinaryinferencecandidate onunchanged88877weights,matched640sourcecontrols; noaccuracygainyet.

V50completed:896improvedOI75.3473/59.1387andoldlitter50.2088/28.7483,butretention93.5/88.3473andfreshlitter36.2952/19.1026regressedfrom640controls99.5/90.7770and49.7143/32.2643. Gatesfailed,originalbenchmarkskipped;incumbent84.0711/62.0606unchanged. Nooptimizersteps.
