# Classifier-only bottle optimization

Prepared after V42 regressed. Restart from protected balanced88877bbf… (84.071068% AP50 /62.060566% AP50–95), not the regressed V42checkpoint.

Training552photos/585boxes/1859exposures, with no source image/old label edits. The88training trash photos received two whole-scene AI reviews:57unchanged positive and4negative photos accepted,27uncertain photos held. One confirmed feather-positive annotation becomes a negative only in a derivative training label. Exclude entire held capture groups and every conservative near-match flag/group, with no overrides. Other retained data preserve V42 ancestry and exposures.

A fixed-seed24photo retention monitor contains16single-bottle photos and8reviewed absence photos. Its source/capture/byte/pixel/rotation-dHash/SIFT aliases are excluded from further training, along with all68older monitor photos. Initial and V42weights have previously trained on the retention parents, so it measures retention and cannot establish unseen generalization or independent NHS accuracy. Sample and reviews froze before any model score.

Only native classifier parameters model.23.cv3.* may train. Feature extractor, neck, box branch and all BatchNorm buffers stay fixed; EMA copies fixed tensors exactly to prevent numerical drift. Real random-input optimizer tests and an actual synthetic2-image Trainer/EMA/saved-checkpoint integration prove decoded box equality and protected state equality. Classification scores update. This does not guarantee unchanged AP50–95 because prediction ranking/NMS membership can still change.

Max6epochs within the user's20epoch cap, AdamW1e-5,b4/nbs16,640,MPS,seed43,mosaic0,scale.25,translate.05,cosineLR,BNstatisticsfixed. Stop after epoch2if either retention score drops more than3points; recover terminal checkpoints without resuming optimizer after a crash.

Final candidates: native retention bestAP50,bestAP50–95 andlatest, at most3distinct trained weights. Require both retention metrics within.5points of initial, all four older-source metrics nonregressing, and mean source gain≥1point. Only then may one candidate receive an original246photo/67box benchmark assessment. Promotion requires both84.071068%/62.060566% floors; move-on requires90% AP50 for that same checkpoint. No validation-label edits, confidence/TTA sweeps, epoch selection on the original benchmark or target guarantees.

Verification:15focused tests pass, including actual synthetic trainer/checkpoint integration and real-runner terminal crash-gap fixtures;237project tests pass. Independent review found and rechecked the repaired terminal-restart bug; no remaining concrete P1/P2. All246benchmark images/labels and protected checkpoint bytes verified unchanged.

Before optimizer steps, the new retention cohort measured99.50% AP50 /90.776984% AP50–95 for initial and99.50% /96.726667% for V42. This does not retrospectively detect V42benchmark regression: bothmodels previously trained on thesephotos, and V42stillincluded them. The new optimizer excludes them and relatedcapture/featurealiases; future retention is a preservationcheck, notproof of unseen generalization. Original benchmark remains the sole90% move-on test.

## Completed six-epoch outcome

Two distinct trained candidates(epoch1and6) passed fixed-state proofs but failed the predeclared litter AP50–95 floor. Epoch1: retention99.50/92.061538, OpenImages77.445216/62.976182, litter48.395074/26.043030(percent AP50/AP95). Epoch6: retention99.50/92.075000, OpenImages83.134033/67.302930, litter49.305893/26.384512. Initial litter47.560575/28.203331. The initial model was correctly retained; no original-benchmark attempt or inference, no promotion or90%claim. Independent read-only outcome review found no receipt/selection inconsistency.
