# Cafeteria object detection

Five YOLO specialists detect eight classes:

| Model | Classes |
|---|---|
| Food/container/carton | Food, hinged container, drink carton |
| Plate/wrapper | Plate, wrapper |
| Utensil | Fork, knife and spoon union |
| Bottle | Disposable plastic drink bottles |
| Packet | Sauce, salt, pepper and sugar sachets |

## Run

```sh
./scripts/run_specialists.sh predict /path/to/image.jpg --device mps --output runs/predictions.json
./scripts/run_specialists.sh evaluate --device mps --output runs/evaluation.json
./scripts/run_specialists.sh train --epochs 40
```

Prediction also accepts a directory. Use `--device cpu` for CPU inference.
Output boxes are normalized `[x1,y1,x2,y2,confidence,class_id]`; class IDs are
listed in `scripts/specialists.py`. The confidence floor is 0.001 for AP;
filter confidence when displaying detections. All detectors use a single pass at 640 pixels,
NMS IoU 0.7, and batch 1. Models load sequentially to limit memory.

## Setup

Requires Python 3.12 and `pip install -r requirements.txt`. Install dependencies
in an environment outside this repo, or use the existing local runtime on this
Mac. Set `CAFETERIA_PYTHON` to select another Python executable.

Place the five pinned checkpoints in `weights/` using the filenames and hashes
in [configs/specialists.json](configs/specialists.json). Training also needs
`weights/yolo11s.pt` (the COCO-pretrained starting checkpoint). Weights are local
assets, supplied separately and excluded from Git.

Datasets, annotations and historical research are stored **outside this repo**:
the sibling `yolo-cafeteria-research-assets/` folder next to this repo. Override with
`CAFETERIA_STORAGE`. Runtime caches default to
`~/Library/Caches/yolo-cafeteria-runtime/`; override with `CAFETERIA_RUNTIME`.
On this Mac the existing data and checkpoints have already been preserved there.

Evaluation defaults to the reviewed manifest under
`artifacts/simple-specialists/shared-eight-reviewed.json` in the external store;
a different manifest can be supplied as a positional argument. Prediction needs
only the checkpoints. Training needs the reviewed V16 packet records in external
`data/packet-focus/broad-v16/` and its prepared runtime source images. A fresh
machine must be supplied those assets; code alone does not recreate the dataset.

## Results

Shared 124-image development evaluation: **84.99% mAP50 / 60.98% mAP50–95**.
These are provisional AI-assisted development results, not independent test
accuracy. See [evaluation notes](docs/evaluation.md) and
[per-class results](results/baseline.json). The later 40-epoch-cap packet candidate
underperformed and was not promoted.

Training separates fitting/monitor groups, uses early stopping and selects by
monitor AP50–95. Only the selected checkpoint receives a final shared-development
evaluation; the active model is preserved. Outputs go to ignored `runs/` and the
external runtime. Keep the laptop lid open; the Mac launcher uses `caffeinate`.

```sh
python -m unittest discover -s tests -v
```

The repo contains maintained code, config, focused tests and concise results.
Datasets and old experiments are no longer committed as the active file tree.
The external store's `history/` folder preserves the previous source, docs and
cleanup audit. Git history is retained; no history rewrite was performed.
