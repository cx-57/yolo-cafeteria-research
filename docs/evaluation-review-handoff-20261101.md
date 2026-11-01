# Evaluation handoff: November 1, 2026

## Completed in the latest research pass

The candidate-screening pool remains **96 visual KEEP candidates**: 52 LeFood-Set food/plates, 15 TrashBox bottles/cartons and 29 Wikimedia utensil/hinged-container/wrapper photos. This is **not** an independently validated cohort. All source/capture model-exposure histories remain unverified; no actual school-bin labels have been checked by a human. Public photos differ from real school cafeteria captures.

A **44-image local annotation bundle** was generated from the two smaller, accessible source ZIPs, deliberately leaving the 52 Mac-only LeFood photos untouched. It includes 44 original image files with SHA-256 and decoded-pixel identities, source and licensing/author metadata, a local browser-based bounding-box review editor, and 27 **unverified AI-draft** object boxes on 14 TrashBox images. The kit includes **all 44 photos**, but three previously identified bottle cases are excluded from potential exploratory scoring because their crowded/ambiguous backgrounds do not permit exhaustive target annotation. Thus **41 are not pre-excluded**, which does *not* make them independently eligible. No original file bytes were changed.

Checks on the 44 packaged photos: zero exact SHA-256 duplicates, zero decoded-pixel duplicates, no dHash pair within radius 5 (minimum distance 10). The 44 have no SHA-prefix matches against the previously logged 100 LeFood-Set screening records. **None of these checks establish historical independence or full provenance.** The ZIP passed CRC and per-image SHA verification. The local annotation export validator passed five synthetic tests; the embedded JavaScript syntax was checked. A browser-level rendering test could not be completed in the hosted environment. The ZIP SHA-256 and screening-exclusion names are preserved in [provisional-screening-20261101.json](../results/provisional-screening-20261101.json).

The artifact named `cafeteria-44-image-review-kit-final.zip` and the companion `cafeteria-review-toolkit-source.zip` were delivered as downloadable conversation artifacts, **not committed to the repository**; the latter contains:
- `scripts/build_review_bundle.py`: assembles original candidate ZIPs and an optional 52-image Mac-local source bundle, validates image hashes, preserves attribution, and embeds the offline editor.
- `tools/review_template.html`: requires per-box human confirmation and explicit school-bin evidence before bin selection.
- `scripts/export_review_labels.py`: checks image identity, rejects incomplete confirmed annotations, and produces human-reviewed class labels. Exploratory bin truth is exported only when every non-excluded image and box has human-entered status and evidence, and is always marked **NOT independent**.
- `tests/test_review_bundle.py`: five synthetic tests.

## Exactly what remains before the independent research experiment

1. Collect new, **sealed school-cafeteria photo sessions** on days never used for training/validation/model selection; log dates and capture groups. Public source images can be supplemental diagnostics, not a substitute for this experiment.
2. Have an annotator verify object boxes for all seven detector classes and independently assign each visible item's correct bin from the *actual* Natick school acceptance policy, recording material, emptiness and other relevant evidence. Resolve uncertainty or mark items for expert review; do not derive truth from the model class.
3. Verify that all new images and capture groups are free of historical training, development, validation, test and checkpoint-selection exposure. The school's own capture logs and the local external asset archive are needed. A hash nonmatch alone is insufficient.
4. Freeze image membership and human labels **before** running the models. Keep all four active model weights, thresholds, and disposal rules unchanged for the first independent benchmark.
5. On the Mac with the frozen checkpoint weights available, run `./scripts/run_specialists.sh sort /absolute/new_photos --device mps --output /absolute/predictions.json` and `python3 scripts/evaluate_sorting.py --predictions /absolute/predictions.json --truth /absolute/human_bin_labels.json --output /absolute/sorting_evaluation.json`. The JSON image paths must match. Only after a valid independent object-label manifest exists should detection mAP be measured using the project's evaluation command.

**Not accomplished:** verified independent field performance, official correct-bin accuracy, actual YOLO inference on the new candidates, or retraining. Those would be false claims without the local model weights and independent human truth. Training on the candidate holdout would contaminate the planned evaluation.
