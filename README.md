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

**Data split verified:** 2,477 original training images, 619 untouched validation images (80/20), and 0 test images. By default, training uses **1× views** (2,477 original image paths), with **randomized YOLO augmentation during training**. There are no redundant hardlinked views. Neither validation nor test is augmented. Grouped duplicates and correlated video images stay together. Train for up to 100 epochs with early stopping.

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

Keep the original split for comparisons. To prepare a **separate** split that conservatively groups the historical audit candidates:

```bash
python3 scripts/train.py --build-only \
  --group-audit data/audits/day1/near_duplicates.csv --output data/training-grouped
```

Use `data/training-grouped/data.yaml` explicitly to train on that new benchmark. Grouping candidates prevents known split risk without deleting photos; the candidates still require visual review.

## Annotation fixes and evaluation integrity

A tracked October 11 review corrected **six visible missing objects in three existing photographs**: two foods and a disposable spoon on LVIS plate scenes, plus two food wrappers in a Trash Detection scene. The preserved plate, food, and bottle annotations remain. The Trash Detection file uses *historical* wrapper class ID 9, which is converted to current ID 8 by the build step. These are manual visual edits, **not** claims that the other images have complete annotations.

**Important:** One corrected image belonged to the previous validation split. The old 73.8% mAP50 figure is **not a comparable baseline for a newly relabeled validation set**. Rebuild the generated split, re-evaluate the *same saved checkpoint* on the updated validation labels, and only then compare a newly trained model against that updated baseline. Never infer an improvement from changing both labels and weights.

GitHub Actions now builds the full original corpus and performs a second fresh label scan. Its checks cover bounding-box geometry, class IDs, source-to-label pairing, split counts and synthetic evaluation edge cases. **Passing those tests does not mean** the annotations are exhaustive or the model detects cafeteria items accurately. Standalone evaluation rejects absent/misaligned AP rows, inconsistent macro averages and NaN scores; unsupported classes are explicitly reported instead of being assigned an invented score.

Run local CPU-level data checks with `python3 -m unittest discover -s tests -v` and `python3 scripts/train.py --build-only`. Running the actual model still requires your local `best.pt` and device; the weights are not in GitHub.

## Source history

The early eight processed sources retain their **old stored class IDs**. `scripts/prepare.py` holds the mapping to the current `config.yaml` IDs. `scripts/train.py` converts IDs **only in the generated training copy**. Newly uploaded food, dry/wet waste and bowl images already use the current IDs. Never point YOLO directly at `data/processed/`; do not change originals.

More details: [data organization](data/README.md) · [source licenses](data/datasets.csv) · [audits](data/audits/).
