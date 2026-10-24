# Bottle monitor campaign V34

User resumed bounded bottle experiments toward90% AP50 and explicitly asked
not to overfit. The separate full-training run remains paused.

Prior best balanced fixed-benchmark pair84.0711% AP50/62.0606% AP50–95;
legacy AP50 leader84.0784%/60.9019%. Keep both protected. Goal remains one
saved/fresh checkpoint>=90% AP50 and>=64% AP50–95. These are development
scores; independent NHS testing is still required for a generalization claim.

To reduce further adaptive use of the fixed246-photo/67-box benchmark,
reserve12 twice-AI-reviewed newly acquired Open Images positive photos
(selectedseed1043 before model measurements) plus24 twice-AI-reviewed new
negative scenes. Monitor36photos/17boxes. Possible generic-pretraining
exposure and small positive support remain limitations; no guarantee against
overfitting is made. Native/fresh monitor scores never count as benchmark
scores. These monitor sources must stay out of future specialist training.

All source/capture groups, byte aliases and decoded-pixel aliases leave the
training list. Source checks covered567current training photos,246benchmark
photos and599additional historical training photos. No exact/SIFT matches;
three dHash flags were visually different photos (Ghadeer vs green crushed
bottle; Rioja closeup vs two school-lunch trays), explicitly resolved before
monitor accuracy measurement. Screening remains heuristic, not proof of full
source independence. No fixed validation label/image/split was changed.

Training567photos/521boxes/2100scheduled exposures. Head-only balancedV19
initialization88877...1518;all BNfixed,AdamW1e-5,batch4/nbs16,640,seed43,
mosaic0,scale0.25,translate0.05. Fixedmaximum9epochs. Keep native monitor
AP50/AP95 winners and latest. Freshly check these plus initialization on the
monitor at ordinary640,batch4,NMS0.5,noTTA. Pick maximum AP50+AP50–95;
ties prefer initial then earliest. Ifinitial wins, keep it without another
benchmark measurement. Otherwise evaluate one distinct chosen candidate on
the original benchmark after training. Declaration/candidate locks persist
across restarts; a monitor score90% cannot trigger an early benchmark.

Checkpoint files and trainer CSV live in the runtime cache outside iCloud;
metadata stays here. Saved final reports interrupted before history are
fully validated and reconciled without inference. Global model promotion
requires improvement without either metric regressing.237project and10
focused checks passed locally, including actual no-inference recovery.
Independent code review found the restartP2, fixed with RED→GREEN tests;
re-review found no new concrete P1/P2. No new benchmark gain is claimed.
