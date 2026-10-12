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

## Set up and train

From your repository folder:

```bash
git pull
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
python3 scripts/train.py --build-only
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
| Review new Roboflow export | `python3 scripts/prepare.py data/raw/school_lunch school_lunch` |
| Import additional LVIS candidate plates | `python3 scripts/lvis.py --help` |

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
