# YOLO Cafeteria Research

**Goal:** detect 11 cafeteria item types with YOLO11n. The model recognizes objects; NHS recycling/compost/trash rules are a separate step.

## Clean layout

```text
config.yaml             # 11 class names and IDs
data/
  processed/            # 3,096 original photos + labels (DO NOT edit)
  raw/                  # historical source archives
  candidates/           # unreviewed images (not training data)
  manifests/            # source/provenance CSVs
  training/             # generated train/validation data (ignored by Git)
scripts/
  train.py              # build 80/20 dataset, train, evaluate
  prepare.py            # convert new Roboflow files into review candidates
  audit.py              # inspect images, labels and provenance
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
| Evaluate best checkpoint | `python3 scripts/train.py --evaluate model/runs/yolo11n-10x/weights/best.pt --device mps` |
| Audit source photos | `python3 scripts/audit.py --manifests data/manifests --output data/audits/current` |
| Review new Roboflow export | `python3 scripts/prepare.py data/raw/school_lunch school_lunch` |
| Import additional LVIS candidate plates | `python3 scripts/lvis.py --help` |

To evaluate on a **genuine test set**, collect and fully annotate **separate NHS cafeteria photos** from different days/cameras. Copy `data/training/data.yaml` to a new local YAML and add `test: /absolute/path/to/nhs_test/images`, with matching labels in a sibling `labels/` folder. Then:

```bash
python3 scripts/train.py data/training/data_with_test.yaml --evaluate model/runs/yolo11n-10x/weights/best.pt --split test --device mps
```

Never use test images to tune the model. Do not report provisional public-source validation scores as real NHS test performance.

## Source history

The early eight processed sources retain their **old stored class IDs**. `scripts/prepare.py` holds the mapping to the current `config.yaml` IDs. `scripts/train.py` converts IDs **only in the generated training copy**. Newly uploaded food, dry/wet waste and bowl images already use the current IDs. Never point YOLO directly at `data/processed/`; do not change originals.

More details: [data organization](data/README.md) · [source licenses](data/datasets.csv) · [audits](data/audits/).
