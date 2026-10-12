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
