# Cafeteria object detection

Four YOLO specialists detect seven classes:

| Model | Classes |
|---|---|
| Food/container/carton | Food, hinged container, drink carton |
| Plate/wrapper | Plate, wrapper |
| Utensil | Fork, knife and spoon union |
| Bottle | Disposable plastic drink bottles |

```sh
./scripts/run_specialists.sh predict /path/to/image.jpg --device mps --output runs/predictions.json
./scripts/run_specialists.sh evaluate --device mps --output runs/evaluation.json
```

Prediction accepts an image or directory. Use `--device cpu` for CPU inference.
Output is normalized `[x1,y1,x2,y2,confidence,class_id]`. IDs 0–6 retain their
existing meanings: food, plate, utensil, hinged_container, drink_carton, wrapper,
bottle. All four models use single-pass 640, NMS IoU 0.7 and batch 1, loading
sequentially. The 0.001 confidence floor supports AP; filter scores for display.

## Setup and assets

Requires Python 3.12 and `pip install -r requirements.txt`. The launcher uses the
existing external environment on this Mac; set `CAFETERIA_PYTHON` to override it.
Place the four pinned checkpoints in `weights/` using the filenames and hashes in
[configs/specialists.json](configs/specialists.json). Weights are supplied separately
and excluded from Git.

Datasets, annotations and historical research live in the sibling
`yolo-cafeteria-research-assets/` folder. Override with `CAFETERIA_STORAGE`.
Runtime caches default to `~/Library/Caches/yolo-cafeteria-runtime/`;
override with `CAFETERIA_RUNTIME`. Prediction only needs the four checkpoints.
Evaluation defaults to external `artifacts/simple-specialists/shared-seven-active.json`;
a custom manifest can be passed as a positional argument.

## Evaluation

The seven retained classes score **88.43% mAP50 / 63.69% mAP50–95** on the same
124 development images. This is the seven-class task after packet removal; model
weights were not improved by dropping a class. These are provisional AI-assisted
development results, not independent cafeteria test accuracy.
See [evaluation notes](docs/evaluation.md) and [per-class results](results/baseline.json).

Condiment packet is retired from inference, evaluation and the active training
entry point. Its code, tests, checkpoints and previous configuration are preserved
externally under `history/packet-retired-20261027/`; earlier experiments and the
rejected long-run candidate remain under `artifacts/packet-focus/`.

```sh
python -B -m unittest discover -s tests -v
```

Only maintained source, configuration, focused tests and concise results are in
Git. Historical source/data/results and Git recovery copies remain outside the
repo. Git commit history is preserved.
