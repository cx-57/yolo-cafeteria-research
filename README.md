# Cafeteria Waste Sorting — Computer Vision + Arduino

Research and prototype code for a cafeteria waste-sorting system. The repository contains **two separate paths**:

1. **Offline research:** four frozen YOLO11 specialist detectors covering food, plate, utensil, hinged container, drink carton, wrapper, and bottle. A separate rules layer produces compost/recycling/manual-review *suggestions*, not verified correct-bin decisions.
2. **Live prototype:** camera detection, a configurable loading-area ROI, Arduino/servo control, and logging. **Detection-only is the default.** Automatic movement is blocked until the ROI, disposal mappings, and physical bin positions are verified.

## Start here

| Task | File or guide |
| --- | --- |
| Live camera and hardware | [Live sorter setup](docs/live-sorter.md) |
| Arduino sketch | [firmware/waste_sorter/waste_sorter.ino](firmware/waste_sorter/waste_sorter.ino) |
| Camera/servo controller | [scripts/live_sorter.py](scripts/live_sorter.py) |
| Offline research inference | [scripts/specialists.py](scripts/specialists.py) |
| Disposal rules | [scripts/disposal.py](scripts/disposal.py) and [configs/disposal.json](configs/disposal.json) |
| Evaluation and audits | [scripts/evaluate_sorting.py](scripts/evaluate_sorting.py), [scripts/research_audit.py](scripts/research_audit.py) |
| Research manuscript | [Research PDF](paper/cafeteria-waste-sorting.pdf) and [source](paper/manuscript.md) |
| Frozen research settings | [configs/research-freeze.json](configs/research-freeze.json) |
| Software verification and limitations | [docs/sorter-verification.md](docs/sorter-verification.md) |
| Real-camera robustness observations | [docs/live-robustness-session.md](docs/live-robustness-session.md) |

## Setup

Python 3.12 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Model weights, original datasets, and private camera footage are **not** tracked by Git. Restore them separately under `../yolo-cafeteria-research-assets/` or configure explicit paths.

### Live controller (safe defaults)

```bash
python scripts/live_sorter.py --image /absolute/path/to/photo.jpg --output runs/sorter/preview.jpg
python scripts/live_sorter.py --probe-cameras 0,1,2
python scripts/live_sorter.py --camera 0 --mode detection
```

See [setup and safety gates](docs/live-sorter.md) before enabling servo movement. The included `configs/sorter.json` deliberately leaves ROI, class destinations, and physical bin assignments **unset**. Do not invent calibration values.

### Frozen research inference

```bash
./scripts/run_specialists.sh predict /absolute/path/to/photo.jpg --device mps --output /tmp/predictions.json
./scripts/run_specialists.sh sort /absolute/path/to/photo.jpg --device mps --output /tmp/suggestions.json
```

## Frozen V1 research metrics

| Metric | Development validation (186 photos) | Historical test (248 photos) |
| --- | ---: | ---: |
| mAP50 | 86.36% | 87.78% |
| mAP50–95 | 65.25% | 67.24% |
| Object precision at confidence 0.50 | 93.00% | 93.42% |

Class-specific thresholds chosen on development validation produced **97.03% precision / 71.43% recall** on validation and **97.22% precision / 68.18% recall** on the historically exposed test set. They have **not** been deployed to the default sorting pipeline.

**Limitations:** These are exposed development cohorts with AI-assisted annotations, not sealed independent tests. Object-detection metrics do not establish disposal-bin accuracy, reliable camera-to-bin performance, or reduced school waste contamination. See [sorting-evaluation protocol](docs/sorting-evaluation.md) and [provenance/readiness report](results/research-readiness.json).

## Repo organization

- `scripts/` — inference, evaluation, audits, experimental training, and live controller.
- `firmware/` — Arduino sorter sketch.
- `configs/` — research freeze, model configuration, and hardware-safe defaults.
- `tests/` — Python and Arduino host-simulation tests.
- `results/` — preserved benchmark summaries and audit receipts.
- `paper/` and `docs/` — manuscript and validation/verification documentation.
- `experiments/` — **historical, non-default** V2–V5 trials, notebooks, and checkpoint metadata. These remain for reproducibility, not because each is part of the live system.

The separately documented V5 seven-class checkpoint stopped after **11/20 epochs** (65.94% validation mAP50; 41.97% mAP50–95). It is provisional, not an independent test result. See [V5 checkpoint record](experiments/yolo-v5-original-plus-v4/V5_11_EPOCH_CHECKPOINT.md). No model weights are shipped in Git.

Do not merge experimental threshold, training-label, or disposal-policy changes into the frozen research results without a new, explicitly identified experiment.
