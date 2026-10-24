# Reviewed full-image bottle trial V31

User requested continued bottle work and to ignore the README workflow.
Goal remains one freshly verified saved checkpoint >=90% AP50 and >=64%
AP50–95. All scores are development model-selection scores on the frozen
246-image/67-box benchmark, not independent test performance.

V30 six-epoch head-only crop trial did not improve baseline. It is archived
with optimizer retained. Fresh selected epoch1 scored83.8031% /61.4286%;
latest epoch6 scored81.7635% /59.0742%. Preserve84.0711% /62.0606% balanced
initialization and84.0784% /60.9019% legacy AP50 leader separately.

V31 adds25 newly reviewed LVIS photos/34 original source boxes. Two batches
(offsets240/300,count60 each,seed42) yielded30/37 usable images. Root AI
reviewed full scenes and enlarged targets; independent second AI review
held three proposed acceptances for unresolved scope/completeness. This
is AI visual review, not human verification or physical material testing.
All67 usable candidates passed the conservative validation overlap screen;
that heuristic is not proof of complete source independence. Original
COCO/Flickr URLs, photo license IDs and LVIS annotation provenance remain
in the downloads ledgers; a code license is not a photo license.

Build from V30’s original full-image exposures with its conservative
training holds, removing derived crop replacements. Final1139 photos,
504 boxes,2056 exposures; new photos repeated4x. No source annotations,
validation labels or splits changed. Source bytes, decoded pixels,
annotation lists, review/screen hashes, exhaustiveness flags, dimensions,
box bounds, all annotation indices and duplicate photos are checked.

Initialize balanced V19 immutable88877...1518. Freeze11 includes the complete
YOLO11 backbone0–10 (including C2PSA); train neck/detection head. All BN
statistics fixed. AdamW1e-5,batch4/nbs16,640,seed43,mosaic0.5,scale0.5,
close_mosaic0;40epoch schedule checked in3epoch phases. This screens the
combined data/capacity intervention and is not an isolated causal ablation.
Live guards reject accidental backbone/BN changes or head-only training.

Reuse V30 immutable fresh evaluation at ordinary640,batch4,NMS0.5,noTTA,
separate AP50/localization/joint/latest candidates, nondominated frontier,
and nonregression-only legacy promotion. Three phases without0.2percentage-
point best-metric progress end this individual screen honestly; overall
user task continues to the next evidence-backed intervention if needed.
237 project tests and11 focused checks pass. Admission and trainability
checks were observed failing before implementation, including a real test
failure when the guard incorrectly allowed backbone block10 to train.
