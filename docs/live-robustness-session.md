# Live robustness work — November 15, 2026

This session began around18:02 America/New_York, at the user's request for
about an hour of continued work. The user required avoiding overfit and later
asked to stop/reverse the live experiment while continuing offline.

**Current state:** camera experiments are stopped. Original trained checkpoints,
datasets, labels, train/validation/test partitions and frozen research policies
were preserved. No training, checkpoint replacement, automatic hardware action
or GitHub push occurred. The saved sorter config remains confidence.60,
ROIunset, verifiedflagsfalse, with ROIzoom disabled by default. The temporary
.25confidence unifiedMPS camera run is no longer active.

## What the actual camera evidence established

The user reported metalspoon/fork and wrapper detections, bread/orange misses,
hand-including food boxes, and a breadcrumb labeledutensil. These are useful
development observations, not an independently scored test cohort.

A lossless saved frame contains a visible metalspoon. The unified checkpoint
predicted **bottle .34702** on it. Offline re-inference on the exact same pixels
reproduced that result on bothCPU andMPS, with nearly identical coordinates and
confidence. The frozen specialist composition predicted **utensil .24728**.
Neither is a sufficiently confident result for automatic sorting at the saved
.60threshold. There is no hardcoded utensil default.

The user's exact breadcrumb and orange failures were not captured with verified
object boxes, so they remain reported errors rather than measured per-object
accuracy. A later saved frame had no testfood/object held up. All diagnostic JSON
files retain `ground_truth: null`; model predictions were not converted into
training truth. Related frames retain one session/group ID.

Local evidence: `runs/sorter/robustness-20261115/offline-spoon-check.json`,
`session-A-isolated/*.png` and matching JSON. These contain local camera imagery;
they are Gitignored and were not published.

## Small-food experiments and decisions

1. A fixed2×2 overlapping food-only tiling experiment was run on the same
   exposed186-image development cohort as the prior comparison. At.60 it
   changed correctfood detections293→296 while falsefood detections rose6→21.
   FoodAP50 dropped94.23%→91.51%. **Rejected; not integrated or enabled.**
2. A size audit found **zero** labeledfood boxes with projected area below32²
   pixels after longest-side resizing to640. The cohort therefore cannot
   establish performance on genuinely tiny foodpieces or crumbs. Its32medium
   and311large foodboxes do not fill that gap. This is a statement about this
   cohort, not every historical image.
3. Added optional **fixed loading-area ROI inference**. The same frozen model
   receives the cropped platform area, with boxes translated back to full-frame
   coordinates. Detections touching the crop edge are marked and cannot become
   eligible for sorting. Full-frame inference remains the default. No ROI was
   guessed, saved or verified on the user's behalf.
4. A deterministic synthetic scale stress check used24food-positive and8food-
   negative sourceimages selected byimageSHA ordering. Entire images, not
   food-dependent boxes, were reduced into a fixed central256square on a
   1280×720canvas; existing annotations were transformed geometrically. The
   fixedROI path recovered many detections lost at the reduced apparent scale.
   **This validates the zoom pathway, not realcrumb accuracy, new generalization
   or independent performance.** No synthetic data was used for training.
5. The already-trained V5 correction ensemble was investigated but its full
   comparison was stopped in favor of the lower-latency work. Completed stage
   caches remain. Its prefix timing includes an intentional process pause and
   resource contention and is not a valid FPS benchmark. No complete accuracy
   or live improvement is claimed for that unfinished trial.
6. A bounded fastparent-six-class-plus-bottle diagnostic was also rejected:
   it lost too much development coverage, particularly wrappers. It was never
   selected or exposed as an active liveprofile.

Receipts/scripts: `runs/sorter/robustness-20261115/small-food-tiles.json`,
`small-food-size-analysis.json`, `roi-scale-check.json`,
`candidate-trial-status.json`, and the corresponding local comparison scripts.
All historical cohorts are exposed development data with provisional labels.
Repeated comparisons must not be called new independent tests.

## Training lineage audit

Wholecitrus and bread are present in both relevant training lineages. For
example, the frozenfood checkpoint's train.txt includes a wholecitrus meal image
`public__school_lunch__010261000_jpg.rf.584af80ad5af6e4fbe3b094b229712f6.jpg`,
with an activeclass0 box over the citrus. The currentunified recipe includes a
related wholecitrus meal image `school_lunch__010261001_jpg.rf.b4167ca47daa172fdb53cec73400ae6a.jpg`.
Bread servings were also found in each actual trainlist. This does not establish
broad coverage of standalone unbagged bread, crumbs or user-camera views.

One important utensil source explicitly retained128black/white disposable
plastic examples from465reviewed images while excludingmetal, transparent,
colored and reusable utensils. This is not a count of diversity across the
whole corpus. Incomplete cross-class annotation and inconsistent historical
chopstick scope are documented in earlier research artifacts.

Historical annotation policies exclude incidental crumbs/rice residue unless a
separate serving is present. Expanding the task to individual crumbs would
require an explicit label-definition decision and suitable evaluation coverage;
threshold changes alone cannot establish that capability.

## Camera and software corrections

- Separate bounded camera startup and detectorwarmup precede serial opening.
  Warmup frames are discarded before stability counting.
- Slowframes in detection-only mode can be inspected as **STALE / diagnostic
  only**. They reset stability and do not rearm. Manual/autonomous modes still
  fail closed at the normal freshness deadline. Camera loss still stops the
  application; hardware does not automatically resume.
- Diagnostics save the exact raw losslessPNG with timing, all low-confidence
  occupancy rows, displayed rows, clipping state, expliciteligibility reason,
  config hash, model hashes, runtime versions and session/group identity.
- Native utensil pooling compatibility reuses the existing contiguous-layout
  adapter; checkpoint bytes and pooling mathematics are preserved.
- A direct combined camera+Torch capture saved a frame but exited139. Its native
  macOS crash report pointed to Apple's CoreML/MLIR model-loading stack. This
  does not prove a particular upstream root cause. Camera-only capture exited0.
  The first raw frame was nearlyblack;45frame settling produced a normal image.
- On macOS, camera-only capture is now isolated in a spawned process without
  Torch. It settles exposure before publishing, drains frames continuously,
  replies only to fresh-frame requests and transports arrays losslessly.
  Startup/read timeouts and childfailure are reported to the parent. Cleanup
  stops a blocked camera child without bringing its native stack into model
  inference. The isolated live run saved correctly exposed rawframes and its
  parent and camera child exited after the user asked to stop.

## Verification and remaining work

Final full repository run: **212 tests, 211 passed and 1 preexisting skip**,
with no failures. Command: `python3 -m unittest discover -s tests -v`.
Log: `/tmp/cafeteria-hour-final-tests.log`. Unit/integration coverage includes ROIcoordinate projection/crop-edge
rejection, stale detection-only vs hardware behavior, matching snapshot/frame
identity, canonical config hashing, warmup-before-serial, rawPNG fidelity,
process capture freshness, blocked-child cleanup and native-child failure
containment. Original weight hashes were checked again unchanged.

This is software and diagnostic progress. **No new learned model accuracy or
physical sorting success is claimed.** Real small-food performance still needs
camera examples collected across multiple foods/materials, scales, backgrounds
and capture sessions. Freeze any selected settings before evaluating a separate
group of physical items. If a failure is used for a change, it becomes
 development evidence and a fresh evaluation group is required.

No training is authorized or performed by this session. Any future training
should be a separate candidate, with diverse reviewed annotations and capture-
group separation, and must preserve the original research and held-out groups.
