# Cafeteria waste-sorting computer vision

Research prototype combining **four specialist YOLO11 detectors** for seven
cafeteria item classes: food, plate, utensil, hinged container, drink carton,
wrapper, and bottle. A separate rule layer suggests compost, recycling, or
manual review. Object recognition is **not** verified disposal-bin accuracy.

## Research paper

**[Read the 10-page PDF research preprint](paper/cafeteria-waste-sorting.pdf)** |
[Editable source and full methodology](paper/manuscript.md)

## Results

| Frozen V1 metric | Development validation (186 photos) | Historical test (248 photos) |
| --- | ---: | ---: |
| mAP50 | 86.36% | 87.78% |
| mAP50–95 | 65.25% | 67.24% |
| Object precision at confidence 0.50 | 93.00% | 93.42% |

With **class-specific thresholds chosen on development validation**, unchanged
V1 weights reached **97.03% precision / 71.43% recall** on validation and
**97.22% precision / 68.18% recall** on the historically exposed test cohort.
These thresholds have **not been deployed** to the default sorting pipeline.
The frozen detector's mAP values did not increase with a confidence cutoff.

**Scientific limitation:** These are exposed development datasets with
AI-assisted labels, not sealed independent tests. The outside-source
41-image exploratory V1 versus V3 comparison also uses approximate AI
annotations. Actual correct-bin accuracy and reduced school waste contamination
have **not been established**.

## Keep these components

| Location | Purpose |
| --- | --- |
| `scripts/specialists.py` | Four-model inference and seven-class object metrics |
| `scripts/disposal.py` | Local-rule suggestions and manual-review logic |
| `scripts/evaluate_sorting.py` | Evaluate against independent, human-verified bin labels |
| `scripts/replay_sorting.py` | Replay decisions from frozen predictions |
| `scripts/precision_sweep.py` | Global threshold diagnostics |
| `scripts/precision_per_class.py` | Validation-only threshold selection, exposed test check |
| `scripts/hinged_candidate.py` | Optional hinged-container V3 experiment |
| `scripts/evaluate_reviewed44.py` | Original versus V3 external-image development comparison |
| `scripts/research_audit.py` | Provenance, annotation quality, and error analysis |
| `scripts/training.py` | Optional isolated training without overwriting frozen weights |
| `configs/` | Active model/rule definitions and research freeze |
| `results/` | Frozen benchmark summaries and research-readiness evidence |
| `docs/` | Paper draft and evaluation/review protocols |
| `tests/` | Tests for retained functionality |

Minor support modules and the reviewed-44 annotation package remain in `scripts/`.
Git does **not** include the large model weights, image datasets, and local
run artifacts. They live in `~/Desktop/yolo-cafeteria-research-assets/`
and are required for actual model inference and reproducibility.

## Investigating real school-photo failures

The [field generalization diagnostic](docs/field-generalization.md) runs the frozen four-detector baseline on new cafeteria photos and saves confidence summaries, possible conflicting detections, and annotated overlays. It **does not** fabricate accuracy without exhaustive human-reviewed labels and does not modify the frozen models. This diagnosis precedes any new unified-model training.

## Running locally on the Mac

```bash
cd ~/Desktop/yolo-cafeteria-research
git pull --ff-only origin main
python3 -m unittest discover -s tests -v

./scripts/run_specialists.sh predict /path/to/photo.jpg --device mps --output /tmp/predictions.json

./scripts/run_specialists.sh sort /path/to/photo.jpg --device mps --output /tmp/suggestions.json

./scripts/run_specialists.sh precision-classwise --device mps \
  --output "$HOME/Desktop/yolo-cafeteria-research-assets/artifacts/classwise-new-run.json"
```

The 44-image offline review kit can still be built from the GitHub Actions
workflow named **GitHub download - 44-image annotation editor and starter JSON**.

## Paper and provenance

- [Research manuscript draft](docs/research-paper.md)
- [Independent sorting evaluation protocol and current limitations](docs/sorting-evaluation.md)
- [V3 / external-photo annotation protocol](docs/reviewed44-v3-scoring.md)
- [Development validation](results/baseline.json) and [historical test](results/test.json)
- [Frozen research configuration](configs/research-freeze.json)
- [Research readiness/provenance audit](results/research-readiness.json)

Every retired workflow, staging script, superseded report, and historical
experiment is preserved in the
[pre-cleanup archive branch](https://github.com/cx-57/yolo-cafeteria-research/tree/archive/pre-cleanup-20261101).

For a publishable field-performance claim, collect fresh sealed school photos,
independently annotate object boxes and verified disposal-bin eligibility, and
evaluate the frozen model **without tuning on those photos**.
