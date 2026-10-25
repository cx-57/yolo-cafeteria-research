# Three-epoch hard-negative classifier logit calibration

User explicitly confirmed: ONLYdisposable plastic drink bottles. Glass, reusable bottles and non-drink containers are not targets. Material is judged from visible appearance and source metadata, not laboratory certification.

V44protected spatial tensors but updated all classifier features; it improved OpenImages yet regressed litter localization and was rejected without an original benchmark measurement. V45returns to balanced88877bbf… and updates only three native final1x1classifier convolutions: six tensors/387parameters. Spatial ANDintermediate classifier features, all BatchNormbuffers and fixed EMA state remain unchanged. Random-input optimizer tests and actual synthetic Trainer/EMA/saved-checkpoint tests verify protection; scores can change and NMS/ranking can still alter AP.

Training-only mining on142reviewedempty-label photos atconf.1found27photos with false proposals. BothAIreviewers approved13of the top16as no eligible drinkbottles: papercups,cigarettes,feather,bag,vape,clearglasswine andnonbeverageketchup. Mixed/uncertain cases7,12,14heldwhole. Each acceptedhardnegative receives8extraexposures; no newindependentparents.

A full122photo scope/completeness review of generic legacypositive sources retained106andheld16entirephotos/groups. Holds include incomplete/mixedannotations and an uncertain reusable-looking sports bottle. Most originals were already plasticdrinkbottles; no confirmednegative conversions were made, and no existing source/benchmark labels changed. Newlyheld sourcegroups and canonicalCOCO aliases areexcluded. The106acceptedphotos have twoAIwhole-scene reviews with nativecrop/originalfollowups; reviewSHAjoins arechecked.

Finaldata533photos/544boxes/1891exposures. Same24photo16objectretentioncohort and68oldermonitorphotos, neverreturnedtotraining. Startingweights previouslysaw retentionparents, so this is preservationmonitoring, not independentNHSgeneralization. Recipe3epochs,AdamW1e-4,b4/nbs16,640,MPS,seed43,mosaic0,scale.25,translate.05,cosineLR. Stopafterepoch2ifretentiondrops>3points; terminalstatesrecoveredbeforetrainingonrestart.

Samepredeclaredgates: eachretentionmetricwithin.5pointsofinitial; allfour older-source APmetricsnonregressing; meanolder-sourcegain≥1point. Atmost3distinctnativecandidates and1gatedoriginalbenchmarkcandidate. Original246photo/67boxlabelsremainunchanged. Promotionneedsboth84.071068%AP50/62.060566%AP95floors;90%AP50move-onmustreferthesamecheckpointontheoriginalbenchmark. Source-onlyscorescannotclaimsuccess.

Resourceincident: the initial mining call passed142filepaths as an in-memory list; the installed Ultralytics loader used listlength as batchsize despitebatch4. Its footprint reached9.5GB; onlythathelperwasstopped, withnodata/checkpointwrites. Explicitfour-imagechunks, MPSmemoryfraction.5 and a regression test fixed it. All142photos then mined successfully; diskspace recovered. Trainingalsohas the MPSmemorycap.

Focused verification18tests passed, including source/mining/review identities, negative-only weights, canonicalaliases, terminalrecovery, actualTrainer/EMA/checkpoint integration and387parameter protection. Independentcode/datareviewfoundnoconcreteP1/P2.
