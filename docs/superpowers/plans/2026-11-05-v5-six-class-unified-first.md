# V5 Six-Class Unified-First Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare a reproducible bottle-free V5 dataset and run real unified-model improvement trials before considering specialists.

**Architecture:** Isolate six-class preparation and trial orchestration from frozen seven-class scripts. Use immutable source/split/run receipts, validation-only selection, and a fail-closed specialist-stage report. Conditional specialist implementation waits for actual completed trials.

**Tech Stack:** Python 3.12, unittest, Pillow, PyYAML, ImageHash, Ultralytics/PyTorch.

**Spec:** `docs/superpowers/specs/2026-11-05-v5-six-class-unified-first-design.md`

## Global Constraints

- Class IDs: 0 food, 1 plate, 2 utensil, 3 hinged_container, 4 drink_carton, 5 wrapper.
- Never substitute the old audit320 dataset for the corrected B187 archive.
- Keep source images unchanged, retain bottle-only negatives, preserve historical weights/reports.
- One fixed group-aware approximately 80/10/10 split; no test evaluation during trials.
- Baseline and U1-U6, then baseline/chosen-candidate repeats at seeds 58/59, before specialist eligibility.
- Do not claim an accuracy gain from tests, dry runs, interrupted runs, or threshold changes.

## Review Focus

- ZIP traversal, duplicate members and stale checksums must fail before writing a derived dataset.
- A source-wide `group` value must not collapse an entire dataset into one meal group.
- Bottle-only photos must survive with empty labels; mixed-class photos keep retained boxes.
- A changed split, training list, image, label or checkpoint must invalidate downstream receipts.
- A failed or truncated run must never count as a completed trial or unlock specialists.

## Task 1: Six-class data preparation

**Files:** Create `scripts/v5_dataset.py`, `tests/test_v5_dataset.py`, `experiments/yolo-v5-original-plus-v4/requirements-six-class.txt`.

**Interfaces:** `read_archive(path, expected_images=2261) -> (rows, contents, source_receipt)` verifies source bytes; `strip_bottle(text) -> (text, removed_count)` validates geometry; `assign_splits(rows, features) -> list[str]` groups and balances; `prepare(archive, output, exclusions) -> dict` emits a self-contained dataset and receipt; `verify_dataset(output) -> dict` verifies every derived file/list and split before training.

- [ ] Write tests for mixed/bottle-only labels, invalid class/geometry, path traversal, checksum mismatch, duplicate entries, group membership, source-group handling, missing class support, exclusion matching, refusal to overwrite and tampering.
- [ ] Run `python -m unittest discover -s tests -p test_v5_dataset.py -v`; observe missing-feature failures.
- [ ] Implement functions and strict production archive/exclusion checks. Record SHA256s for source, labels, images, class counts, split groups, lists and configuration. Group exact/perceptual duplicates and explicit meal IDs; retain earlier holdouts outside training.
- [ ] Run the focused tests and commit the passing implementation.

## Task 2: Unified trial orchestration

**Files:** Create `scripts/v5_trials.py`, `tests/test_v5_trials.py`.

**Interfaces:** consumes `verify_dataset`; `trial_recipe(trial, seed) -> dict` defines baseline/U1-U6; `run_trial(dataset, output, trial, seed, device, batch, smoke=False) -> dict` trains and validates, writing status/epoch/class/checkpoint receipts; `stage_report(dataset, runs) -> dict` inventories completed trials and prevents premature specialist work.

- [ ] Write tests proving recipe budgets, bounded train-only balancing, six-class model checks, incomplete/failed/smoke/mismatched/tampered receipts rejected, finite metrics required, held-out evaluation excluded, and baseline/chosen-candidate seeds 57/58/59 required.
- [ ] Run `python -m unittest discover -s tests -p test_v5_trials.py -v`; observe expected failures.
- [ ] Implement immutable run configuration, failure receipts and actual validation metrics. U6 combines supported choices or extends convergence when none helps. Save operating-point curves and predictions for later group-bootstrap comparisons; the stage report cannot certify eligibility without completed repeats and uncertainty analysis.
- [ ] Verify focused tests and commit. Do not implement specialists before measured eligibility.

## Task 3: CLI, notebook and real-data handoff

**Files:** Create `scripts/v5_experiment.py`, `experiments/yolo-v5-original-plus-v4/SIX_CLASS_UNIFIED_FIRST.md`, and a six-class Colab notebook; modify README with links only.

**Interfaces:** CLI `audit`, `prepare`, `plan`, `train`, `status`; notebook calls the same CLI, so it cannot silently diverge in class maps or recipes.

- [ ] Exercise CLI missing-source and existing-output errors; notebook compile check and focused integration test use controlled ZIPs.
- [ ] Document corrected archive and explicit 28-image exclusions, commands, dependency setup, resumption limits, artifact storage and specialist gates.
- [ ] Run full unittest suite, CLI help, source audit and real preparation if the corrected archive becomes available. Review the entire change with a fresh reviewer.
- [ ] Start actual baseline only after a valid corrected dataset, exclusions and fixed split exist. Record blocked inputs precisely if unavailable; do not fabricate a run or use the old archive.

## Execution ledger

2026-11-05: User said continue after the concrete design and missing-data report; proceed inline on `codex/v5-six-class-unified-first`. No extra worktree is needed: the dedicated branch already separates these changes from main, and the current checkout is clean. The corrected file remains absent at the provided path. Preparation/training on it remains pending; code and controlled-fixture verification can proceed.

Task 1: complete — 11 focused data-preparation tests passed; committed as cb2c87b4.

Ruling: The user subsequently offered to run training epochs and requested code. Deliver a standalone Colab notebook and runnable CLI; do not start actual experiment training locally. Full preparation/training and specialist/uncertainty decisions remain pending returned real data/results.

Task 2: implementation complete — recipe, provenance, balancing, stage-gate and actual inference-only validation-adapter checks pass. No epoch-based accuracy result is claimed. Specialist eligibility intentionally remains false pending returned results and group-bootstrap analysis.

Task 3: handoff implemented — notebook embeds maintained modules and can execute without publishing a branch. Whole-branch review found resume fallback, failure-receipt ordering, environment mismatch and label-cache bypass defects; regression tests reproduced them and fixes preserve phase-aware recovery and frozen-label loading. Full experiment execution remains pending the user's GPU runs.

2026-11-06 Ruling: The user requested local training and then explicitly authorized the available audit320 ZIP after being told it lacks later corrections. Add an explicit source-authorization flag and record the older revision/missing corrections. Do not require or fabricate the later B187-specific exclusion identities for the older archive. Source and held-out provenance checks still apply. Local training is now authorized, replacing the earlier user-run handoff assumption.

2026-11-06 local preflight: MPS available, but only approximately 485 MiB disk space initially free. Added a pre-output disk-space check; no experiment training started. User was asked to free at least 5 GB or provide an external-drive path. Audit and in-memory split verification can proceed without materializing the image dataset or allocating checkpoints.

2026-11-06 Ruling: User authorized cleanup of unnecessary files while preserving important data. Removed specifically identified browser, updater-download, package and compiler caches; saved a cleanup receipt outside Git. Preserved research data/weights, browser profiles/offline storage, active pending VS Code update and Android SDKs. Reclaimed approximately 4.26 GB. On the 8-GB Mac use batch 1 and a 3-GiB startup reserve plus a live per-batch 1-GiB guard instead of the initial static 5-GiB threshold; this permits a measured local attempt while guarding checkpoint/swap space. Keep batch override consistent across candidates.

2026-11-06 Ruling: User changed the immediate request to 10 local epochs. Gracefully interrupted the 40-epoch campaign during epoch 1 before any checkpoint existed, preserving its failure receipt and log. Added a validated per-run epoch override and launched only `baseline_10epoch_seed57` with `--epochs 10 --batch 1 --device mps`; no further variants queued. It starts from general pretrained weights because the partial original epoch had no checkpoint. A completed 10-epoch run is a pilot and does not satisfy the full-trial specialist gate. Full suite: 132 tests run, one skipped, all others passed.
