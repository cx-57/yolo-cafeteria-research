# V5 without bottle: unified trials before specialists

This is a new six-class experiment. The old seven-class V5 checkpoint and V1 specialist system are preserved. Nothing here establishes an accuracy improvement until full runs have completed and been compared.

Open **YOLO_V5_Six_Class_Unified_First.ipynb** in Colab, select a GPU runtime, and run cells in order. It contains the three Python scripts, so it works without publishing a branch or downloading code from a guessed GitHub URL. The notebook's source snapshot is regenerated from the CLI modules when this handoff is packaged.

## Inputs

1. `YOLO_V5_CommandCenter_B187_QC_selected_merged1572.zip` — the corrected, complete 2,261-image V5 archive. The older audit320 ZIP is refused by `prepare`. An audit does not train anything.
2. The **28-image ambiguity exclusion ledger** documented in `ANNOTATION_AUDIT_STATUS.md`. Supply a CSV with an `image_path` column containing the 28 exact paths from `v5_original_plus_v4/manifest.json`, or a JSON list of those paths. If the delivered ledger uses another column, map it using the assignment/decision records; do not guess matches or just exclude the first 28 images. This is a required input, not an empty placeholder. The original worker CSV may already be inside the corrected ZIP; the notebook lists likely ledger filenames for inspection.

The corrected ZIP is still absent from the local workspace at handoff. The exclusions were also not available for inspection. Code tests use controlled fixtures. All original annotations remain provisional, including retained background negatives; filtering bottle does not certify other labels are complete.

**2026-11-06 local source change:** The user explicitly authorized the available older `YOLO_V5_partial_annotation_audit320(5).zip`. For that exact archive, `prepare --allow-provisional-audit320` records the older revision and missing later corrections. The later B187 28-image ledger is not assumed to apply or fabricated. This explicit mode does not change the default corrected-archive workflow. Local preparation/training requires at least 5 GiB free for the derived data, checkpoints and swap.

```bash
python scripts/v5_experiment.py prepare \
  --archive "$HOME/Downloads/YOLO_V5_partial_annotation_audit320(5).zip" \
  --allow-provisional-audit320 \
  --output "$HOME/Desktop/yolo-cafeteria-research-assets/artifacts/v5-audit320-six-class-20261106/data"
```

## What changes

Derived names are `[food, plate, utensil, hinged_container, drink_carton, wrapper]`, with unchanged IDs 0–5. Preparation removes bottle annotation rows, retains mixed images and bottle-only background images, and hashes every output. The new detector is trained with six outputs from general pretrained weights. It does not merely hide bottle predictions from the old checkpoint.

Preparation verifies original manifest hashes, groups exact and conservative perceptual duplicates and explicit meal/session IDs, keeps previous held-out images outside training, and builds a fixed approximately 80/10/10 split. Ambiguous exclusions are applied first, so totals will be less than 2,261. Each retained class must have train, validation and test support; failed grouping or inadequate support stops preparation. Heuristic grouping cannot prove all camera sessions are independent.

## Run from a terminal

From the repository root, using an appropriate Python environment:

```bash
python -m pip install -r experiments/yolo-v5-original-plus-v4/requirements-six-class.txt
python scripts/v5_experiment.py audit \
  --archive "$HOME/Downloads/YOLO_V5_CommandCenter_B187_QC_selected_merged1572.zip"
python scripts/v5_experiment.py prepare \
  --archive "$HOME/Downloads/YOLO_V5_CommandCenter_B187_QC_selected_merged1572.zip" \
  --exclusions /absolute/path/exclusions_28.csv \
  --output /absolute/path/v5-six-class/data
python scripts/v5_experiment.py train \
  --dataset /absolute/path/v5-six-class/data \
  --output /absolute/path/v5-six-class/runs/baseline_seed57 \
  --trial baseline --device mps
```

On Colab, use `--device 0`. A baseline runs 40 epochs. GPU time depends on the runtime; use the baseline epoch timings before deciding how much of the full campaign to run in a session. Nothing automatically purchases compute.

After the baseline, run all remaining trials and confirmation seeds:

```bash
python scripts/v5_experiment.py campaign \
  --dataset /absolute/path/v5-six-class/data \
  --runs /absolute/path/v5-six-class/runs --device 0
python scripts/v5_experiment.py status \
  --dataset /absolute/path/v5-six-class/data \
  --runs /absolute/path/v5-six-class/runs
```

The campaign skips only previously verified complete trial/seed pairs. Each output directory belongs to exactly one run. An interrupted run with a recorded failure and matching `last.pt`, input hashes, environment and configuration can be resumed using the identical `train` arguments plus `--resume`; the campaign attempts this automatically. It checks that the checkpoint retains an epoch and optimizer state before resuming. An abrupt runtime kill may leave no failure receipt, so automatic resume then refuses. Preserve the folder and return it for recovery; do not rename unrelated weights into it or repeat training over existing files. If training completed and only evaluation failed, `--resume` recovers evaluation from the verified `best.pt` without running additional training epochs. Do not move prepared datasets after creation: their hashed lists contain absolute paths.

Use a stable persistent directory (the notebook uses Google Drive) so Colab resets do not erase data and checkpoints. Only `best.pt` and `last.pt` are retained as standard checkpoints, avoiding a large extra checkpoint for every epoch. The runner discards derived label caches before training and validation so cached labels cannot bypass the frozen text-label checks.

## Experiment sequence

| Run | Epochs | Change |
| --- | ---: | --- |
| baseline | 40 | Fresh pretrained YOLO11s, 800 pixels, batch 4 |
| U1 | 40 | Less scale/mosaic augmentation |
| U2 | 40 | 1024 pixels, batch 2 |
| U3 | 40 | Up to 3× training exposure for baseline-weak classes |
| U4 | 40 | Lower learning rate |
| U5 | 40 | Pretrained YOLO11m, batch 2 |
| U6 | 40 or 80 | Combine supported changes, or extend baseline convergence |
| confirmations | 40 or 80 each | Repeat baseline and selected candidate at seeds 58 and 59 |

All candidates use the same fixed validation photos. No reserved-test score selects a model. Logs include actual completed epochs, environment, per-class AP50/AP50–95, recall at a validation-selected precision of at least 80%, threshold curves, per-image matching statistics, prediction labels, validation plots, timings and SHA256 receipts. An unattainable precision target has recall 0 with a null threshold. A two-epoch `--smoke` run is diagnostic only and cannot count toward specialist eligibility.

## Return these results before specialists

Run the notebook's export cell to create `V5_Six_Class_Unified_Results.zip`, containing source/split provenance, run receipts, configs, logs, curves, per-image validation stats and validation overlays. Checkpoints remain in the persistent run folders; return them too when needed for inference or further work.

The stage report always keeps `specialists_allowed=false`: this implementation deliberately stops at `uncertainty_analysis_required` after valid trials and repeats. Return the ZIP for actual paired group-bootstrap analysis and label/error review. The source design's proposed targets are AP50 ≥0.80 and recall ≥0.70 at precision ≥0.80, with proposed useful gains of +0.02 AP50–95 or +0.05 recall and a 0.02 AP50–95 regression guard for other classes. Tiny class cohorts may require more reviewed validation data before a conclusion.

Specialists will be implemented only for classes still lacking after that evidence. If unified training improves them, keep improving the unified path. A specialist's whole-pipeline benefit and added latency must be measured before adoption.
