# Experimental hinged-container improvement — separate model version

**Goal:** improve open foam clamshell/takeout-container recognition while preserving food and carton detection. **Frozen research baseline is not changed.**

## Why this experiment

On the 44-photo public-source visual stress test, only four of twelve
`hinged_container_candidate` source-hint photos contained any corresponding
detector output. This is **not 33% measured recall**: the images have no human
object-level ground truth. Most open clamshells were instead recognized as
plates or food. The frozen validation set has only **two** hinged-container
ground-truth boxes (versus four on exposed development test); a high AP score
there is insufficient evidence of real-world reliability.

The hinged class belongs to the three-class `food-container-carton` YOLO11n
specialist, not the separate plate/wrapper specialist. This training recipe
**preserves the original three-class checkpoint head**; it does not accidentally
reinitialize the class head as seven-class or train on the 44 public stress photos.

## Training design

- Source: existing **training partition only** from
  `yolo-cafeteria-research-assets/data/combined-seven-school-oct29-20261031/data.yaml`.
  The script checks exact image overlap with the existing validation/test splits.
- Class remapping: old class IDs `food=0, hinged_container=3, drink_carton=4`
  become `food=0, hinged_container=1, drink_carton=2` for the original
  three-class checkpoint. Other objects are deliberately unlabeled
  *background* in this specialist-only experiment.
- Include every existing training image with a hinged-container annotation.
  Up to 250 additional carton-positive and 250 food-positive photos, plus
  up to 90 **plate-only negative** photos, are selected with a fixed seed.
- Repeat each hinged-positive **training view** three times; this is
  oversampling, **not three times more independent photographs**.
  Native YOLO train-time transforms supply mild rotation, translation, scale,
  shear, color/lighting and horizontal flips. No copied image or annotation
  from validation, test, the 44-photo review ZIP, or a sealed future field
  holdout is used for training.
- Initialize from the exact SHA-pinned `food-container-carton.pt`;
  run eight epochs (user-adjustable), AdamW with LR 0.00005, batch 1,
  frozen first ten layers, and 640-pixel resolution.
- Save all generated images/labels, sample records, training logs and the
  independent **candidate** `best.pt` under the external assets folder.
  Compare weights before/after to prove active checkpoints remain unchanged.
- Compare candidate vs frozen baseline using the **same** 186-photo repaired
  validation manifest, never the 248-photo exposed test for model selection.
  Require per-class and aggregate AP50/AP50–95 not to regress by more than
  0.5 percentage points; no automatic promotion, regardless of gate result.
- Run a separate 44-image source-hint-only *qualitative* comparison. Emit
  side-by-side overlays for all 12 hinged-source photos and the matching
  detection count on those 12, plus carton-source counts. These source
  fractions are not recall/accuracy.

## Run these commands on the Mac

From the project root, update Git and run:

```bash
cd ~/Desktop/yolo-cafeteria-research
git pull --ff-only origin main

CANDIDATE="$HOME/Desktop/yolo-cafeteria-research-assets/artifacts/hinged-candidate-v2"

./scripts/run_specialists.sh hinge train --out "$CANDIDATE" --device mps
./scripts/run_specialists.sh hinge evaluate --out "$CANDIDATE" --device mps
./scripts/run_specialists.sh hinge stress --out "$CANDIDATE" --device mps \
  --bundle "$HOME/Downloads/cafeteria-44-image-review-kit-github.zip"
```

The `train` command refuses any preexisting output folder; use a fresh
versioned directory for each new experiment. If training completes but a
subsequent evaluation fails, **do not repeat training**: troubleshoot the
`evaluate` or `stress` command separately. The output is stored outside Git,
where the original source photos and checkpoints live.

## What to send back

The two small result files:

`$CANDIDATE/validation-gate.json` and
`$CANDIDATE/stress-comparison/comparison.json`

The comparison also includes side-by-side overlays in
`$CANDIDATE/stress-comparison/hinged-overlays/`. They show actual changes to
the detector, including possible regressions. **Do not replace the active model
or claim improved accuracy just because the candidate found more source-hint
objects.**

As a first step, even a promising candidate would remain *development only*.
A future independent, human-labeled school capture session is required for a
credible claim of new generalization or correct-bin sorting accuracy.
