# YOLO Cafeteria Research

Research repository for evaluating whether a real-time YOLO-based computer vision system can provide accurate and reliable waste-disposal guidance under realistic school cafeteria conditions.

## Research question

> **To what extent can a real-time YOLO-based computer vision system provide accurate and reliable disposal guidance under realistic school cafeteria conditions?**

## System

```text
camera
  ↓
YOLO object detector
  ↓
physical item class
  ↓
local disposal-rule engine
  ↓
compost / recycling / trash / manual check
```

The detector predicts **physical objects**, not disposal streams. Disposal policy is kept in a separate rules file so the model can remain unchanged when local waste rules change.

## Repository layout

- `configs/` — class taxonomy, disposal rules, dataset definitions, and experiment configs
- `data/` — manifests and local dataset layout; raw/processed image data are not committed
- `src/data/` — dataset audit, filtering, remapping, deduplication, validation, and split tools
- `src/training/` — model training entry point
- `src/evaluation/` — detection, disposal, robustness, error, and latency evaluation
- `src/inference/` — detector and real-time disposal-guidance pipeline
- `experiments/` — compact experiment ledger
- `results/` — paper-ready metrics, figures, tables, and qualitative examples
- `docs/` — research protocol, taxonomy, annotation rules, and deployment protocol
- `paper/` — manuscript and references
- `tests/` — integrity checks for mappings and dataset structure

## Data principle

Public data are filtered by **visual/domain match**, not just class name. A ceramic restaurant plate, for example, should not count as useful training data for a disposable cafeteria plate simply because both are labeled `plate`.

Raw images, processed datasets, model weights, and Ultralytics run directories are intentionally excluded from Git.

## Status

Scaffolding initialized. The next research milestone is to freeze the NHS cafeteria object taxonomy after a real waste audit and populate `data/manifests/` with strictly matched sources.
