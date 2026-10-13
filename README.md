# YOLO Cafeteria Research

**Goal:** detect 11 cafeteria item types with YOLO11n. The model recognizes objects; NHS recycling/compost/trash rules are a separate step.

## Clean layout

```text
config.yaml             # 11 class names and IDs
data/
  processed/            # 3,096 original photos + audited labels (track every label repair in Git)
  raw/                  # historical source archives
  candidates/           # unreviewed images (not training data)
  manifests/            # source/provenance CSVs
  training/             # generated train/validation data (ignored by Git)
scripts/
  train.py              # build 80/20 dataset, train, evaluate
  prepare.py            # convert new Roboflow files into review candidates
  audit.py              # inspect images, labels and provenance
  diagnose.py           # class support, training curves, fixed-validation ablations
  lvis.py               # import/download LVIS plate candidates
model/
  weights/              # downloaded pretrained YOLO weights
  runs/                 # trained best.pt, plots and metrics
tests/                  # automated checks
```

Only `model/README.md` is committed: the model weights and training outputs are created after running training.

## V2 redesign (research and experiments)

The [V2 research-backed plan](docs/REDESIGN_V2.md) defines wrapper versus bottle versus condiment labels, a target-domain NHS photo protocol, reproducible class-weighted YOLO11n versus YOLO11s experiments, and validation rules. The existing training pipeline now supports optional `--cls-pw 0.25` and `--freeze 10`; these are **off by default**. No V2 checkpoint has been trained and no NHS test score has been claimed. Retain V1 outputs unchanged.

## Implemented V2 workflow

[Step-by-step V2 commands](docs/V2_IMPLEMENTATION.md) are now available. The
[single V2 workflow](scripts/v2.py) creates separate audited dataset copies,
imports signed-off NHS photos grouped by session, applies checksum-verified
human-approved label replacements, fingerprints the build, and launches three
fixed training trials (YOLO11n, weighted YOLO11n, YOLO11s).

**Strict V2 validation is NHS-only; a test set is optional.** A strict build
requires independently reviewed NHS validation photos and all-11-class
positive support in training and validation (and test, if supplied). Without
those NHS validation photos, use the provisional builder on the existing
80/20 split for exploratory training:

    python3 scripts/v2.py build --provisional

The provisional builder moves Roboflow near-identical source groups or exact
photo duplicates that cross train/validation into **V2 train copies only**,
records each change in `v2_build.json`, and leaves V1 untouched. This can
change V2 validation membership; do not compare its AP directly to V1.

Tests: `python3 -m unittest discover -s tests -p 'test_v2.py' -v`.
The old checkpoint and dataset remain unchanged. No V2 accuracy improvement
has been measured.

## Codex-reviewed V2 data and tests (October 12)

The [Codex review completion/status audit](docs/V2_CODEX_STATUS_20261012.md)
records the exact checklist, what passed on the full 3,096-image GitHub
dataset, and what still needs independent cafeteria photos or human review.

To **actually use** the 84 reviewed image-level label replacements (including 20 tightened wrapper boxes verified Oct 13) and 242 LVIS food
review records in a new V2 dataset (without modifying V1):

```bash
python3 scripts/v2.py build --source data/training \
  --output data/training-v2-codex-reviewed \
  --provisional --use-codex-review
```

Five additional paper-boat candidate images are available with provisional
AI-assisted boxes but **not human verified**. To use the drafts in
TRAINING ONLY for a separate research experiment, add:
`--candidate-manifest data/v2_candidates/lvis_trays/provisional_manifest.csv`
and select a different output directory. For the expanded **16-photo**
train-only candidate set (5 LVIS paper-boat photos + 11 licensed Commons
plates, wrappers, bottles, cartons, packets, napkins and bowls), use:
`--candidate-manifest data/v2_candidates/provisional_train_20261013.csv`.
The 38 proposed boxes are **AI-assisted and not yet human-verified**.
GitHub Actions generates a contact sheet for full-resolution annotation review.
Strict V2 and validation refuse unverified candidates. The separate test
set remains missing; paper boats, napkins and condiment packets have
**zero public validation objects**, and bowls have only two. The evaluator
now warns if any class has under 30 validation instances; it does not
claim that 30 suffices for research-quality precision. No model accuracy
improvement has yet been established by a controlled training comparison.

See [38-overlay root-cause audit and fixes](docs/V2_WEAK_CLASS_OVERLAY_AUDIT_20261013.md).
The `inspect_predictions.py --all-images --no-overlays` mode diagnoses all
591 validation photos without creating hundreds of JPEGs. Fresh training
requires pinned Ultralytics 8.4.175 (or explicit `--allow-unpinned`).

**October 13 weak-class experiment:** The existing `scripts/experiment.py`
now supports `--crop-repeats` for training-only 60% crops centered on small
plate/wrapper/bottle targets. Use the frozen reviewed V2 source and compare
a bounded 2× repeat control to the same-size crop arm on the *same*
591-image validation split. Crops remap all intersecting annotations;
GitHub CI validates the full real-corpus build and unchanged validation bytes.
See [paired commands and evaluation rules](docs/V2_IMPLEMENTATION.md#weak-class-small-object-intervention-october-13-matched-ab-test).
This remains experimental: no claim of 80% recall or an NHS test score.

## Set up and train

From your repository folder:

```bash
git pull
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
# Only generate data/training when it does NOT already exist; rebuilding replaces it.
python3 scripts/train.py data/training/data.yaml --device mps
```

Use `--device mps` for Apple Silicon, `--device 0` for an NVIDIA GPU, or omit it for automatic device selection.

**Data split verified:** 2,477 original training images, 619 untouched validation images (80/20), and 0 test images. By default, training uses **1× views** (2,477 original image paths), with **randomized YOLO augmentation during training**. There are no redundant hardlinked views. Neither validation nor test is augmented. Grouped duplicates, tracked historical near-duplicate candidates, and correlated video images stay together by default. Grouping near-duplicates is conservative: it does not prove two images are identical. Train for up to 100 epochs with early stopping.

Important: some newer sources remain training-only to prevent source leakage, so validation is not representative of every class. Paper boats have no labeled training photos, and paper boats/napkins/condiment packets/bowls lack provisional validation examples. The 344 LVIS plates stay in `data/processed/plate/` and are included provisionally.

## Commands

| Task | Command |
|---|---|
| Prepare the dataset without training | `python3 scripts/train.py --build-only` |
| Train a model | `python3 scripts/train.py data/training/data.yaml --device mps` |
| Train with 1× views (default) | `python3 scripts/train.py --epochs 80 --batch 4 --device mps` |
| Resume after interruption | `python3 scripts/train.py --resume model/runs/yolo11n-1x/weights/last.pt` |
| Evaluate best checkpoint | `python3 scripts/train.py --evaluate model/runs/yolo11n-1x/weights/best.pt --device mps` |
| Audit source photos | `python3 scripts/audit.py --manifests data/manifests --output data/audits/current` |
| Diagnose model/data limits | `python3 scripts/diagnose.py --run model/runs/yolo11n-1x --output data/audits/model_plateau_2026-10-11` |
| Compare crop vs full-frame detection | `python3 scripts/crop_diagnostic.py --device mps` |
| Compare 640px vs 960px inference | `python3 scripts/crop_diagnostic.py --device mps --compare-imgsz 960` |
| Diagnose cached errors + export review overlays | `python3 scripts/analyze_cached.py --review-overlays 15` |
| Review new Roboflow export | `python3 scripts/prepare.py data/raw/school_lunch school_lunch` |
| Import additional LVIS candidate plates | `python3 scripts/lvis.py --help` |

The [October 12 extended investigation](data/audits/model_revisions_2026-10-12/investigation.md) summarizes measured failures, research, and a controlled next-training plan. An audit flagged **60 potentially redundant ingredient boxes in 7 LVIS images** for human inspection ([review queue](data/audits/model_revisions_2026-10-12/ingredient_review.csv)); none have been deleted. `scripts/analyze_cached.py` reuses the existing JSONL inference caches and reports overlap-aware one-to-one recall, per-class/source/size breakdowns, a ranked unmatched-detection CSV and optional four-panel image comparisons. It does not train or relabel. The initial 55-image cohort is not an NHS test set.

The resolution comparison reuses the same 55 validation images and existing 640px cached predictions in `model/runs/crop-diagnostic/progress.jsonl`. The additional 960px inference resumes from `resolution_960_progress.jsonl` and writes `resolution_960_summary.json`. It reports recall at IoU 0.50, extra unmatched predictions, and the effect of class-aware suppression of overlapping crop boxes. **Unmatched detections are not confirmed false positives** because the annotations are incomplete; no trained checkpoint or images are uploaded to GitHub.

To evaluate on a **genuine test set**, collect and fully annotate **separate NHS cafeteria photos** from different days/cameras. Copy `data/training/data.yaml` to a new local YAML and add `test: /absolute/path/to/nhs_test/images`, with matching labels in a sibling `labels/` folder. Then:

```bash
python3 scripts/train.py data/training/data_with_test.yaml --evaluate model/runs/yolo11n-1x/weights/best.pt --split test --device mps
```

Never use test images to tune the model. Do not report provisional public-source validation scores as real NHS test performance.

## Investigating the 74% result

The saved best checkpoint at epoch 40 reports **73.8% mAP50, 50.3% mAP50–95, 82.8% precision, and 72.1% recall**. These are detection metrics, not classification accuracy. The run reached epoch 42 of 80 planned, and the best localization score improved at epoch 40.

The [performance investigation](data/audits/model_plateau_2026-10-11/README.md) documents confirmed missing annotations in mixed plate/food scenes, missing validation coverage for four classes, and 28 historical near-duplicate candidate pairs crossing the current split. It includes visual evidence, a review queue, and controlled fine-tuning/source-ablation commands.

Evaluation now writes per-class precision, recall, F1, AP, target support, and explicit unsupported status in `metrics.json`. Use `--imgsz` for a resolution experiment; without it, standalone evaluation uses the checkpoint's size. `--optimizer AdamW --lr0 0.0001 --cos-lr --mosaic 0` enables a low-rate no-mosaic fine-tuning experiment from `--model path/to/best.pt`. These controls enable comparisons; improved performance must be measured.

The historical ungrouped split should only be used to reproduce older experiments. **New builds automatically group** the tracked historical near-duplicate candidates; this changes which photos may be held out. To prepare an explicitly named grouped copy:

```bash
python3 scripts/train.py --build-only \
  --group-audit data/audits/day1/near_duplicates.csv --output data/training-grouped
```

Use `data/training-grouped/data.yaml` explicitly to train on that new benchmark. Grouping candidates prevents known split risk without deleting photos; the candidates still require visual review.

## Annotation fixes and evaluation integrity

A tracked October 11 review corrected **six visible missing objects in three existing photographs**: two foods and a disposable spoon on LVIS plate scenes, plus two food wrappers in a Trash Detection scene. The preserved plate, food, and bottle annotations remain. The Trash Detection file uses *historical* wrapper class ID 9, which is converted to current ID 8 by the build step. These are manual visual edits, **not** claims that the other images have complete annotations.

**Important:** Default rebuilds now also group tracked near-duplicate candidates, so **both the validation membership and at least one validation annotation can differ from the old run**. One corrected image belonged to the previous validation split. The old 73.8% mAP50 figure is **not a comparable baseline for a newly relabeled validation set**. Re-evaluating the old checkpoint on the new labels is only **diagnostic**: because the split changed, that checkpoint may previously have trained on some newly held-out images. For a fair experiment, train both a new baseline and the proposed improvement **from the same pretrained weights on the same corrected, grouped split**, with the same data, seed and evaluation protocol. Compare their supported per-class metrics, then assess a fully annotated independent NHS test set. Never infer improvement from simultaneously changing labels, split and weights.

GitHub Actions now builds the full original corpus and performs a second fresh label scan. Its checks cover bounding-box geometry, class IDs, source-to-label pairing, split counts and synthetic evaluation edge cases. **Passing those tests does not mean** the annotations are exhaustive or the model detects cafeteria items accurately. Standalone evaluation rejects absent/misaligned AP rows, inconsistent macro averages and NaN scores; unsupported classes are explicitly reported instead of being assigned an invented score.

Run local CPU-level data checks with `python3 -m unittest discover -s tests -v` and `python3 scripts/train.py --build-only`. Running the actual model still requires your local `best.pt` and device; the weights are not in GitHub.

## October 12: corrected annotation experiment and actual prediction review

The newer [source-level diagnosis](data/audits/README.md) identifies poor plate, wrapper and bottle AP, and the absence of napkin, bowl, condiment and paper-boat validation. **Do not delete occluded plates merely because they are difficult.** Their existing boxes are kept; exposure to partly visible objects matters for cafeteria detection.

Codex added `scripts/lvis_food.py`, `scripts/experiment.py` and `scripts/partial_labels.py`. The reviewed official LVIS food overlay proposes 1,364 boxes across 242 plate-source photos; the full GitHub Actions dataset check accepted 1,355 additions and skipped 9 overlapping existing labels. The *experiment* preserves all 2,477 original training and 619 validation photos and all existing plate boxes; with selected 2× class repetition it builds 3,208 training views. Those views are repetitions, not new photos.

To prepare that **separate, non-destructive** dataset:

```bash
python3 scripts/train.py --build-only
python3 scripts/experiment.py --source data/training --output data/training-revised \
  --label-additions data/audits/model_revisions_2026-10-12/lvis_food_additions.json \
  --balance-classes plate wrapper bottle --max-repeat 2
```

**Do not compare the earlier 77.9% mAP50 directly against numbers from this experiment.** Correcting 33 validation LVIS photos changes the ground truth. The script keeps the images in the same split, but not the validation labels. The official LVIS food annotations also are not proof of exhaustive 11-class annotations.

### Inspect the saved 100-epoch model on real images (without retraining)

```bash
python3 scripts/inspect_predictions.py \
  --weights model/runs/yolo11n-corrected-baseline-100/weights/best.pt \
  --data data/training/data.yaml --split val \
  --classes plate wrapper bottle napkin --per-class 4 \
  --output model/prediction-review
```

The script creates side-by-side JPEGs (**green: ground truth; red: model predictions**) and `summary.json`, showing model boxes on up to four images per requested class. Napkins have **no** validation targets, so zero will be selected for napkins on `--split val`. To visually inspect napkins from training (not an accuracy estimate), repeat with `--split train --classes napkin --output model/prediction-review-train`.

An unmatched predicted box can be a genuine missed annotation, not necessarily a hallucination. The overlay's fixed confidence threshold and IoU matches are **diagnostics**, not Ultralytics AP or an NHS test. The user's `best.pt` and real validation dataset are local and are not committed to GitHub, so GitHub Actions cannot measure this model's actual error rate.

### Source-aware experimental loss (only after inspection)

Codex's experimental `source_coverage.json` marks which classes each source is expected to annotate. `--coverage-policy` now connects the opt-in loss through `scripts/train.py`. It avoids treating negatives for *unknown* source classes as reliable background, while retaining labeled positive supervision. This is an experimental hypothesis, **not a proven improvement**. It requires Ultralytics **8.4.175** and `--mosaic 0` (also no MixUp/CutMix/copy-paste). Do not resume an ordinary checkpoint into this loss.

When ready for a controlled comparison, train **both** a standard and masked model from the **same pretrained checkpoint**, on `data/training-revised/data.yaml`, with otherwise identical hyperparameters and fixed validation data. Example short preliminary runs:

```bash
python3 scripts/train.py data/training-revised/data.yaml --model yolo11n.pt \
  --epochs 10 --batch 4 --device mps --workers 0 --mosaic 0 \
  --name corrected-standard-test
python3 scripts/train.py data/training-revised/data.yaml --model yolo11n.pt \
  --epochs 10 --batch 4 --device mps --workers 0 --mosaic 0 \
  --coverage-policy data/audits/model_revisions_2026-10-12/source_coverage.json \
  --name corrected-masked-test
```

Only run these after model prediction review and after CPU integration tests pass. A 10-epoch comparison is a **smoke/feasibility experiment**, not a definitive performance claim. Neither model has independent NHS validation coverage for all 11 classes; additional capture-group-disjoint labeled scenes and paper boats remain necessary.

## Source history

The early eight processed sources retain their **old stored class IDs**. `scripts/prepare.py` holds the mapping to the current `config.yaml` IDs. `scripts/train.py` converts IDs **only in the generated training copy**. Newly uploaded food, dry/wet waste and bowl images already use the current IDs. Never point YOLO directly at `data/processed/`; do not change originals.

More details: [data organization](data/README.md) · [source licenses](data/datasets.csv) · [audits](data/audits/).
