# Bottle box projection implementation plan

> Execute inline using superpowers:executing-plans, under user's ongoing optimization authorization and20epoch ceiling.

**Goal:** Test geometry adaptation with exact score/feature retention, after V45/V46 score adaptations failed main benchmark transfer.

**Architecture:** Start from balanced88877 incumbent. Train only final1x1cv2 box-distribution projections (six tensors, actual count verified); all classifier features/logits, backbone/neck, intermediate regression features and BN buffers remain bitwise fixed, including EMA saved checkpoints. Add only V47 photos accepted by both AI reviews and conservative overlap/source-group checks to unchanged V45 training schedule. No old labels, monitors or benchmark change.

**Recipe:** Max3epochs,AdamW1e-5,b4/nbs16,640,MPS,seed43,mosaic0,scale.25,translate.05,cosineLR,warmup1. Same retention stop: afterepoch2 drop>.03 in either metric stops. Save nativeAP50/localization/latest candidates; terminal recovery forbids more epochs after stop/budget. Up to3distinct saved candidates on unchanged92development monitor photos; atmost1original246/67benchmark candidate if same V45source gates pass. Original two-metric nonregression floors84.071068%/62.060566%,90%AP50move-on. No independent test claim.

**Alternative considered:** Additional score blends are rejected: V46alpha.5source gains regressed to83.3384/61.0726 on main benchmark. Broad feature tuning regressed inV42. Final box projections test localization directly while avoiding the previously observed confidence drift. These choices are implementation judgments, not user promises of improvement.

## Tasks

- [ ] Verify V47source/dual-review/overlap bindings; derive additive dataset from immutableV45 records; prove no monitor/capturegroup/byte/pixel/flagged aliases admitted. Bind complete source boxes, photo geometry, SHA, review and source metadata; no per-box repair.
- [ ] Add failing regression-projection tests: exactly6mutable tensors, real optimizer changes decoded boxes but class scores exactly unchanged, EMA restoration, protected drift rejection. Implement nativecv2terminal selection and real Trainer/EMA/save smoke. Reuse tested optimizer lifecycle and predeclared gate policy.
- [ ] Freeze data, code, source ancestry and recipe before training; run focused/project tests and independent review. Execute3epochs; verify exact saved checkpoint hashes, frozen benchmark files and actual protected tensors. Archive results and push code/evidence; never promote a regression.

**Review focus:** Code-import collisions in reused experiment ancestry; unbound new source labels; source split/group/near aliases; classifier tensor drift under EMA/serialization; crash at final epoch/stop marker before completion receipt. Existing terminal tests plus actual integration checks cover recovery.
