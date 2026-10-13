# YOLO V2: implemented workflow

V2 code is in scripts/v2.py and reuses scripts/train.py and
scripts/experiment.py. Original V1 datasets and checkpoints are not modified.
No new trained model or NHS accuracy gain is being claimed.

## Step 1 — start with a safe provisional build

From the repository directory on the Mac:

    git pull
    python3 -m pip install -r requirements.txt
    python3 -m unittest discover -s tests -p 'test_v2.py' -v
    python3 -m unittest discover -s tests -p 'test_experiment.py' -v
    python3 scripts/v2.py build --source data/training --output data/training-v2 --provisional

Use the EXISTING data/training/data.yaml as your frozen V1 split. Do not run
train.py --build-only if data/training already exists: that command deletes
and regenerates the generated V1 split before writing a new one. If the
source is missing, first back up your V1 split metadata and checkpoint and
then rebuild it intentionally.

The V2 builder now checks for legacy Roboflow exports of the SAME source
photograph across train/validation. If it finds cross-split groups or exact
image duplicates, it moves the implicated VALIDATION COPIES into V2 training,
including transitive duplicate groups. It never changes the original V1
images, labels, or split. Every move appears in v2_build.json under
legacy_split_repairs and legacy_validation_images_moved. If this would empty
validation, the builder fails instead of training without validation.

**Important:** these fixes mean the V2 validation split may have FEWER than the
original 619 photos, and its mAP must not be compared directly to the old V1
score. Reevaluate/retrain competing models on the same fixed V2 cohort.

This produces a V2 copy with an auditable JSON summary and support counts.
Because it contains public-source validation and lacks several classes, it is
NOT a valid NHS evaluation. The developer must not label it as production-ready.

Optional separate provisional LVIS food overlay, still requiring review of
nested food-portion/ingredient candidates:

    python3 scripts/v2.py build --source data/training --output data/training-v2-lvis --provisional --label-additions data/audits/model_revisions_2026-10-12/lvis_food_additions.json

Output directories must be NEW. There is no automatic overwrite.

## Step 2 — approve corrections to original errors

A local reviewed_overrides.json consists of objects like this:

    [
      {
        "source": "trash_detection",
        "image": "train__000055_jpg.rf.29dacaafe4c7e56a1bae80e830784b02.jpg",
        "expected_label_sha256": "REPLACE_WITH_HASH_OF_ORIGINAL_LABEL",
        "complete": "yes",
        "reviewer": "reviewer_id",
        "reason": "Manually verified flexible food wrapper, not a bottle",
        "rows": ["8 0.50 0.50 0.40 0.25"]
      }
    ]

These coordinates are format examples, NOT actual verified boxes. Replace
the rows with the COMPLETE human-reviewed label file, including all objects
from all 11 classes in the image. An empty list is valid only for a truly
reviewed negative image. Find the original corresponding label in the
data/training/labels/{train,val}/ directory, then use the Mac command:

    shasum -a 256 path/to/old_label.txt

Pass the override path to the build command with --reviewed-overrides.
The builder checks the reviewer, complete=yes, legal geometry, exact source
filename and original label hash, then changes only V2 copied labels.

## Step 3 — photograph and annotate real NHS cafeteria still scenes

Keep all local photographs and labels under data/nhs/, which Git ignores.
Use the 11 numeric IDs and class definitions in config.yaml and
docs/REDESIGN_V2.md. Take photos at different distances, lighting conditions,
camera angles and cafeteria backgrounds, with wrappers, bottles, cartons,
condiment sachets, paper boats, plates, food, etc. No videos are required.
Avoid capturing identifiable bystanders without permission.

Create data/nhs/manifest.csv with this header and REAL records:

    image,label,split,session,reviewer,reviewed,complete
    day1/meal01.jpg,day1/meal01.txt,train,2026-10-12-morning,reviewer_a,yes,yes
    day2/meal02.jpg,day2/meal02.txt,val,2026-10-13-lunch,reviewer_a,yes,yes
    day3/meal03.jpg,day3/meal03.txt,test,2026-10-14-lunch,reviewer_a,yes,yes

These are example rows, not existing photographs. CSV image/label paths are
relative to the manifest directory. Each annotation file contains standard
YOLO lines with class_id center_x center_y width height, normalized to 0..1.
A blank label file is allowed for a fully reviewed negative image.
Mark reviewed=yes and complete=yes only after every visible object of
ALL eleven classes has been considered.

All photos from one capture session must use the SAME split. The importer
rejects session leakage, exact duplicates, missing labels and missing signoff.
For strict mode, train and NHS validation must each contain positive
instances of all eleven categories. A test set is **optional**; if you later
add one, it must likewise contain all eleven classes. Counts alone do not
guarantee semantic completeness.

## Step 4 — construct strict V2 (NHS-only evaluation)

Once the reviewed NHS dataset exists:

    python3 scripts/v2.py build --source data/training --output data/training-v2-nhs --nhs-manifest data/nhs/manifest.csv --reviewed-overrides path/to/reviewed_overrides.json

Omit the --reviewed-overrides argument if there are no signed-off corrections.

STRICT MODE: legacy public V1 train/validation images become training-only
COPIES in V2. New V2 validation (and optional test) contains ONLY reviewed
NHS scenes, grouped by capture session. Missing training/validation class
support blocks strict builds. Without a test set, V2 reports development
validation metrics, not independent final test performance. All V1 files stay
untouched. The new V2 has a complete file fingerprint in v2_build.json.
It cannot independently certify that reviewers labeled every visible object.

## Step 5 — controlled training (same prepared data, all else held constant)

Run a two-epoch integration test if desired:

    python3 scripts/v2.py train --data data/training-v2-nhs/data.yaml --trial control-n --epochs 2 --device mps

Then three full comparisons:

    python3 scripts/v2.py train --data data/training-v2-nhs/data.yaml --trial control-n --epochs 80 --device mps
    python3 scripts/v2.py train --data data/training-v2-nhs/data.yaml --trial weighted-n --epochs 80 --device mps
    python3 scripts/v2.py train --data data/training-v2-nhs/data.yaml --trial control-s --epochs 80 --device mps

All trials use seed 42, 640px, 80 epochs, batch 4, AdamW, LR 0.0007,
cosine LR, mosaic 0.3 and workers 0. Weighted-n adds only cls_pw=0.25.
Control-s changes only YOLO11n to YOLO11s. A custom --tag sets a distinct
run suffix; outputs are not silently overwritten. Add --dry-run to inspect
commands without launching Ultralytics. Build fingerprint is checked each run.

Only for NONFINAL exploratory model experiments on the earlier provisional
data, explicitly add --allow-provisional.

### Investigate missing-label suppression (paired trial; optional)

Some source datasets label only one or two of the 11 classes. Normal YOLO loss
penalizes detections of real, **unlabeled** classes as background. V2 now
exposes a **paired experiment** to test whether suppressing *unknown-class
negative classification loss* improves plate / bottle / wrapper recall.

Both commands use the **same frozen V2 dataset, pretrained YOLO11n model,
training settings, 1× image views, and zero mosaic**. Only the experimental
loss differs. Keep the 80-epoch ordinary `control-n` run intact.

```bash
python3 scripts/v2.py train \
  --data data/training-v2-codex-reviewed/data.yaml \
  --trial no-mosaic-n --epochs 80 --batch 4 --device mps \
  --allow-provisional --tag coverage-comparison

python3 scripts/v2.py train \
  --data data/training-v2-codex-reviewed/data.yaml \
  --trial coverage-n --epochs 80 --batch 4 --device mps \
  --allow-provisional --tag coverage-comparison

python3 scripts/v2.py report \
  --trials no-mosaic-n coverage-n --epochs 80 \
  --tag coverage-comparison --allow-incomplete
```

For a quick integration run before committing to 80 epochs, use a new tag
and `--epochs 2` for **both** trials. This tests feasibility, not accuracy.
The source coverage policy is in
`data/audits/model_revisions_2026-10-12/source_coverage.json`. Trials reject
unknown training sources instead of silently assigning wrong supervision.
The masked loss uses pinned Ultralytics 8.4.175; mosaic, MixUp, CutMix and
copy-paste must remain disabled to preserve photo provenance. The standard
training recipes are unchanged.

**Limitations:** Source-level coverage flags are provisional. Even a known
class may have missing instances. This experiment cannot add missing boxes,
generate paper-boat photographs, or guarantee an AP improvement. The public
validation split still lacks four classes and is not an NHS test set. If
performance is unchanged/worse, use actual held-out prediction reviews and
new verified NHS images rather than repeating an unsuccessful training recipe.

### Faster weak-class-only fine-tuning (without forgetting strong classes)

If food, cartons and hinged containers already perform well, do **not**
train a fresh 3-class detection head. The existing checkpoint predicts
**all 11 classes**, so changing its class count or stripping labels would
break comparability and cause forgetting. Instead, prioritize photographs
containing plate, wrapper, or bottle and replay **15% of the remaining
original strong-only photographs**. Each original and its paired focus
crop stay together. This creates just a path list and YAML, no new
image copies; original labels, V1/V2 images, and all 591 validation
images remain unchanged.

After both previous 3,295-view datasets have been built:

```bash
git pull
python3 scripts/experiment.py \
  --source data/training-v2-weak-focus \
  --output data/training-v2-weak-fast \
  --weak-subset --replay-fraction 0.15
```

Inspect the printed view count and `weak_subset.json` before training.
It will be **smaller than 3,295**, but the exact count depends on the
actual labeled objects. Train using the best **existing full V2** checkpoint:

```bash
WEIGHTS=model/runs/v2-control-n-reviewed-baseline/weights/best.pt
test -f "$WEIGHTS" || { echo "Find your correct checkpoint with: find model/runs -name best.pt"; exit 1; }
python3 scripts/train.py data/training-v2-weak-fast/data.yaml \
  --model "$WEIGHTS" --name v2-weak-fast10 \
  --eval-name v2-weak-fast10-eval \
  --epochs 10 --batch 4 --device mps --imgsz 640 --workers 0 \
  --optimizer AdamW --lr0 0.0001 --lrf 0.1 --cos-lr \
  --mosaic 0 --warmup-epochs 0 --patience 0
```

Compare `model/runs/v2-weak-fast10-eval/metrics.json` to the **same frozen
validation** results from the original checkpoint, especially plate,
wrapper and bottle *and* food, drink carton and hinged container recall/AP.
Only a matched baseline experiment can isolate crop-specific gains. A
smaller training subset saves steps per epoch; it is not a guarantee of
higher recall. No class is permanently frozen. Strong-class regression
means more replay or a return to full-dataset training may be necessary.

### Weak-class small-object intervention (October 13; matched A/B test)

The prior 55-photo diagnostic matched **6/26 plates** and **3/20 bottles**
whose box area was under 1% of the image (confidence .25, IoU >= .50).
It also recorded high-confidence bottle/wrapper class confusion. Enlarging
small object views is a **training hypothesis**, not a measured improvement.

Use the latest **already reviewed** V2 directory as source so no outdated
annotations are cropped. These two builds use the same originals and byte-
identical validation images/labels. Both add at most one extra training view
for a training photo containing plate, wrapper or bottle; the first repeats
the full image and the second crops around a small target when feasible.
The same 2× cap applies to BOTH. All original train photos stay.

```bash
git pull
python3 scripts/experiment.py --source data/training-v2-codex-reviewed \
  --output data/training-v2-repeat-control \
  --balance-classes wrapper bottle plate --max-repeat 2
python3 scripts/experiment.py --source data/training-v2-codex-reviewed \
  --output data/training-v2-weak-focus \
  --balance-classes wrapper bottle plate --max-repeat 2 --crop-repeats
```

From the same existing **full V2 pretrained-finetuned checkpoint** (edit the
weights path if your run used another tag), use an equal 15-epoch budget and
otherwise identical training settings. The new images use 60% crops when a
target occupies under 4% of the original; all labels are clipped/remapped.
If an image lacks a valid crop target, its extra view remains a full-image
repeat, so the experiment does not fabricate new data.

```bash
WEIGHTS=model/runs/v2-control-n-reviewed-baseline/weights/best.pt
python3 scripts/train.py data/training-v2-repeat-control/data.yaml \
  --model "$WEIGHTS" --name v2-repeat-ft15 --eval-name v2-repeat-ft15-eval \
  --epochs 15 --batch 4 --device mps --imgsz 640 --workers 0 \
  --optimizer AdamW --lr0 0.0001 --lrf 0.1 --cos-lr \
  --mosaic 0 --warmup-epochs 0 --patience 0

python3 scripts/train.py data/training-v2-weak-focus/data.yaml \
  --model "$WEIGHTS" --name v2-focus-ft15 --eval-name v2-focus-ft15-eval \
  --epochs 15 --batch 4 --device mps --imgsz 640 --workers 0 \
  --optimizer AdamW --lr0 0.0001 --lrf 0.1 --cos-lr \
  --mosaic 0 --warmup-epochs 0 --patience 0
```

Read per-class `recall`, `ap50`, `map50_95` and `support` from
`model/runs/v2-{repeat,focus}-ft15-eval/metrics.json`. Evaluate wrapper,
bottle and plate alongside food/carton regressions. A brief fine-tune is a
diagnostic, not a conclusive final benchmark. The training corpus still lacks
new independent NHS wrapper/plate/bottle photos. If this comparison fails,
more repetitions or arbitrary larger input size are **not** justified.

## Step 6 — compare

    python3 scripts/v2.py report --trials control-n weighted-n control-s --epochs 80

The report verifies identical V2 dataset hashes and class coverage before
ranking mAP50-95, plus mAP50, precision and recall. Partial class coverage
fails unless --allow-incomplete is explicitly specified for diagnosis.
Inspect per-class metrics, wrapper-vs-bottle confusion and Mac latency before
deployment. Select on NHS validation. A future independent NHS test, if
collected, should be evaluated ONCE for the final selected model.

## Limitations

- The current repo has no new reviewed NHS photos or NHS test set.
- **A test set is not required to start V2 training or validation.** Until
  NHS validation exists, use --provisional for existing 80/20 data and
  --allow-provisional only for explicitly exploratory training.
- No V2 checkpoints have yet been trained.
- Missing paper boats cannot be fixed with synthetic augmentation alone.
- Current annotation review queues are candidates, not approved corrections.
- The builder and automated tests enforce file/label/split rules; they cannot
  guarantee visual annotation quality.
