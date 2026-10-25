# Prepared full bottle training — not launched

The user requested90% validated AP50 before moving on, without overfitting, and earlier asked to pause before full training. This directory is preparation only until that pause is explicitly lifted.

Prepared dataset:725unique source photos,700bottle boxes,2732scheduled training exposures. It preserves all603V36parents/561boxes/2244exposures, adding58OpenImages positives,29fresh litter positives and35reviewed absence backgrounds. New source repeats are2/8/4respectively; repeats are not independent photos. All68permanent monitor photos, byte/pixel aliases and available source groups are excluded. Source boxes remain unchanged and full frames are used.

Recipe:40epochs maximum with patience10, all YOLO11s features trainable from balanced88877bbf…, BatchNorm running statistics fixed, AdamW1e-5,batch4/nbs16,640,MPS,seed43,mosaic.3,scale.5,translate.1,close_mosaic10,cosine LR. This is a hypothesis to test, not a predicted90% result. Previous small-data short trials regressed; V40 also showed poor localization on fresh litter.

Native checkpoint retention uses the combined68photo/35box monitor. Fresh final selection evaluates at most3distinct trained checkpoints on both cohorts separately, ranks minimum AP change then mean change versus the balanced initialization, and keeps the initial model eligible. All four source metrics must be nonregressing and mean improvement≥.01 before one original benchmark candidate may be evaluated. The original benchmark cannot choose epochs.

A promotion requires both original benchmark AP metrics to meet or exceed84.071068% /62.060566% for the same checkpoint. Move-on eligibility requires AP50≥90% plus that nonregression;64–70% AP50–95 remains the stated localization aspiration. Monitor scores cannot satisfy move-on. Development cohorts are not an independent NHS test.

The runner checks a declaration-bound authorization receipt before any training. No authorization or training-start file has been created. At least2GiBfree is required. APFS byte-identical copy-on-write sharing recovered space while retaining every checkpoint/image/native-library file and independent inodes; storage receipts record byte hashes.

Verification:12focused V42tests,18V40tests and237root tests pass. Independent read-only review found no concrete P1/P2 findings, including final declaration/preflight hashes. Actual runner invocation without authorization exited at the pause guard before dataset traversal/training. All246original benchmark image/label hashes and both protected checkpoint SHAs match. Remote CI is not verified; prior account billing/spending failure remains external.
