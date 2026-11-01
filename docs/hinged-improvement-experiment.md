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
  The script checks exact image overlap with the existing validation/test splits and with every SHA-verified photo in the 44-image diagnostic ZIP. A matching training photo blocks the experiment. The 186 training-time validation images use the corrected labels from the pinned frozen validation manifest, not the stale disk labels.
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

./scripts/run_specialists.sh hinge all --out "$CANDIDATE" --device mps \
  --bundle "$HOME/Downloads/cafeteria-44-image-review-kit-github.zip"
```

The `all` command trains, runs the frozen 186-image validation comparison, and creates 12 side-by-side hinged-source image overlays in one sequence. It refuses any preexisting output folder; use a fresh versioned directory for each new experiment. If training completes but evaluation or comparison fails, **do not repeat training**: separately rerun `./scripts/run_specialists.sh hinge evaluate --out "$CANDIDATE"` or `./scripts/run_specialists.sh hinge stress --out "$CANDIDATE" --bundle "$HOME/Downloads/cafeteria-44-image-review-kit-github.zip"`. The output is stored outside Git,
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


## V4 recovery run (2026-11-01): gently fine-tune V3 without touching frozen models

V3 had 87.39% overall AP50 on the previously evaluated, historically exposed **248-photo test**, vs 87.78% frozen. V3 found the hinged class on 8/12 source-hint photos versus 4/12 frozen, but these 44 public photos have no verified object boxes, and the development validation has only TWO hinged objects. The 248-photo test and repeatedly inspected 44-photo diagnostics must **not** be used to tune V4.

V4 tries to recover the food and carton regressions by **starting from V3's SHA-verified `best.pt`**, reducing the learning rate from `0.00005` to `0.00002`, reducing eight to **four epochs**, and reducing the hinged-image oversampling factor from **three to one**. The 29 AI-visual draft-labeled new photos remain training-only. There is no guarantee of improved AP.

On the user's Mac (same paths used for V3):

```bash
cd ~/Desktop/yolo-cafeteria-research
git pull --ff-only origin main

./scripts/run_specialists.sh hinge all \
  --parent "$HOME/Desktop/yolo-cafeteria-research-assets/artifacts/hinged-candidate-v3-retry" \
  --out "$HOME/Desktop/yolo-cafeteria-research-assets/artifacts/hinged-candidate-v4" \
  --device mps \
  --epochs 4 \
  --repeat 1 \
  --lr 0.00002 \
  --bundle "$HOME/Downloads/cafeteria-44-image-review-kit-github.zip" \
  --extra-annotated "$HOME/Downloads/hinged-29-annotated-train-only-AI-DRAFT.zip"
```

Preflight checks block wrong/missing/tampered V3 weights, altered specialist class mappings, altered non-target specialists, wrong original checkpoint provenance, and different validation cohorts. The same exact-byte training-vs-validation/test/stress anti-overlap checks apply. Historical near-duplicate leakage is still unverified.

After completion, inspect:

- `hinged-candidate-v4/candidate-validation.json`: V4's corrected 186-photo development validation mAP50 and per-class scores.
- `hinged-candidate-v4/validation-gate.json`: retention against frozen original (not a promotion command).
- `hinged-candidate-v4/parent-validation-comparison.json`: V4 versus V3 on the same corrected validation cohort.
- `hinged-candidate-v4/stress-comparison/comparison.json` and `parent-stress-comparison.json`: V4 vs frozen and V4 vs V3 **source-hint detections**, not mAP or verified recall.

The **248-photo test** should only be compared after model selection on the validation data; no automatic test runs or checkpoint activation. If V4 regresses, keep the original frozen model and V3 as an experimental checkpoint.
