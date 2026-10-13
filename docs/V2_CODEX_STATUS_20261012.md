# V2 Codex completion audit and verified test runs

**Date:** 2026-10-12. The checklist below distinguishes implemented code, mechanically
verified repository data, visually reviewed label proposals, and tasks that
cannot be finished without additional independent images or a full training run.
Neither the old model nor V1 labels have been overwritten.

## Checklist: actual status

- [x] **Mechanical audit of all 3,096 original images.** The tracked audit
  inventories 6,493 pre-revision labeled boxes and class coverage. This checks
  file/label structure, not visual completeness of 3,096 photographs.
- [ ] **Exhaustive visual sign-off for every object in every image.**
  Contact sheets and exception queues exist, but the review log
  data/audits/v2_review_2026-10-12/image_review.csv is empty; this task
  CANNOT truthfully be marked complete yet.
- [x] **Implement 11 reviewed image-level annotation corrections in V2 only.**
  Codex correction file: data/audits/v2_review_2026-10-12/reviewed_overrides.json.
  This includes a non-bottle cup, missing utensils, and duplicated bowls.
- [x] **Review the 242 LVIS food-overlay records.** Codex reduced the original
  1,364 suggested boxes to 605 approved boxes through portion-level review,
  including the nested-ingredient conflicts; actual full 3,096-image CI build
  accepts 603 and skips 2 overlapping with existing same-class boxes.
  **Food-only review is not 11-class completeness certification.**
- [ ] **Resolve all wrapper-versus-bottle confusions.** Some individual
  labeling problems were reviewed, but no exhaustive repaired count or measured
  post-retraining improvement is established.
- [~] **Expand scarce paper-boat data.** Five previously existing LVIS tray
  photos now have AI-assisted provisional YOLO labels containing 7 paper-boat
  boxes and 13 other target boxes. They are explicit **unverified drafts**.
  Import requires --provisional and --candidate-manifest; strict or validation
  import is rejected. Human visual approval is still needed.
- [ ] **Increase diverse real cafeteria images for wrappers, paper boats,
  napkins, condiments, bowls and plates.** No new NHS-specific photos were
  supplied; multiple classes still have weak original support.
- [x] **Rebuild a leak-repaired V2 copy without altering V1.** GitHub Actions
  built 2,505 training / 591 validation originals from the same 3,096
  images; it moved 28 leaking validation copies (18 connected image groups)
  into V2 training only.
- [ ] **All 11 classes evaluated on independently reviewed cafeteria
  validation.** In this provisional V2 validation, paper_boat, napkin,
  condiment_packet and bowl remain unsupported. The separate test set is
  optional and is not present.
- [x] **Freeze V2 image/label content with SHA-256 dataset fingerprint.**
  The V2 builder and training wrapper refuse changed data after a build.
- [ ] **Complete full 80–100 epoch YOLO11n training on corrected V2.**
  This requires the full local dataset and Mac/GPU, not this CI smoke.
- [~] **Test class weighting and YOLO11s.** The opt-in cls_pw=0.25 and
  yolo11s.pt workflows exist; short end-to-end integration runs are used to
  validate functionality. Full matched-budget training comparisons remain open.
- [ ] **Select a winner using paired class-wise evaluation and speed.**
  Requires complete training and fixed, sufficiently covered validation.
  No improved real-world accuracy is claimed.

## Confirmed automated tests

- [GitHub V2 test workflow](https://github.com/cx-57/yolo-cafeteria-research/actions/workflows/v2-tests.yml):
  real 19+ synthetic V2 unit/integration tests have passed on GitHub.
- [Full-corpus build with reviewed labels](https://github.com/cx-57/yolo-cafeteria-research/actions/runs/38325972020):
  the 3,096-image Codex-reviewed V2 builder passed, with 603 added LVIS food
  boxes, 11 reviewed replacement labels, and leakage repairs.
- [Real end-to-end YOLO smokes](https://github.com/cx-57/yolo-cafeteria-research/actions/runs/38326190504):
  pinned Ultralytics CPU successfully fit pretrained YOLO11n with and without
  cls_pw, generated validation metrics, saved and reopened checkpoints,
  and performed inference. These were each **1 epoch at 128px on a tiny
  selected subset**; both diagnostic mAP50 values were about 0.0422.
  **These are NOT evidence of competitive detection performance.**
- Additional full-corpus candidate import and YOLO11s smoke checks live in
  .github/workflows/pipeline-tests.yml; consult the actual workflow run
  conclusion rather than assuming pending jobs passed.

## Build a genuinely changed, experimental V2 dataset NOW

From the repo root on your Mac, after verifying the existing V1 split has
not been regenerated:

    git pull
    python3 -m unittest discover -s tests -p 'test_v2.py' -v
    python3 scripts/v2.py build --source data/training \
      --output data/training-v2-codex-reviewed \
      --provisional --use-codex-review

This uses all 11 approved label overrides plus the 242 reviewed LVIS records.
It does NOT add new photos. To also add the **unverified provisional
paper-boat drafts** to training ONLY, select a DIFFERENT new output directory:

    python3 scripts/v2.py build --source data/training \
      --output data/training-v2-paperboat-drafts \
      --provisional --use-codex-review \
      --candidate-manifest data/v2_candidates/lvis_trays/provisional_manifest.csv

The draft photos and their proposed labels are in
data/v2_candidates/lvis_trays/ and provisional_labels/. Inspect each image and
its YOLO boxes carefully. **A draft label is not a finished annotation.**
Source image rights and suitability should be checked before public release.

Inspect the JSON report in the selected output directory:
v2_build.json, experiment_summary.json, and data.yaml. Keep all compared
models on the same frozen selected directory.

An optional short engineering run:

    python3 scripts/v2.py train \
      --data data/training-v2-codex-reviewed/data.yaml \
      --trial control-n --epochs 2 --device mps \
      --allow-provisional --tag codex-review-smoke

## External blockers: not code bugs

The repository does not contain newly photographed, independently annotated
NHS cafeteria scenes with all 11 categories in validation; the "final test"
is likewise absent but **not required** for development. Only the user or
someone with access to the cafeteria and a labeling workflow can collect those
real-world photographs or approve every proposed box. The trained 100-epoch V1
checkpoint and full fresh V2 training logs live locally rather than in GitHub.

Do NOT check off full accuracy improvement, complete validation support, all
3,096 images visually reviewed, or independent deployment quality without
relevant evidence.
