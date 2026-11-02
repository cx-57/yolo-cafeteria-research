# Diagnose the unseen cafeteria failure

The 87.78% historical mAP50 came from previously exposed development images, not an untouched school test. Do not retune on the historical test.

## Run the frozen detectors on the Mac

Use the **existing** four specialists on the exact school photos where detection failed. Save outputs locally, not in public Git.

~~~bash
cd ~/Desktop/yolo-cafeteria-research
git fetch origin
git switch --track origin/experiment/field-generalization-audit-20261102

./scripts/run_specialists.sh field \
  --photos "$HOME/Desktop/cafeteria-failed-photos" \
  --device mps \
  --output "$HOME/Desktop/yolo-cafeteria-research-assets/artifacts/field-baseline-20261102"
~~~

The tool requires existing local model weights. The output directory must not exist yet.

Outputs:
- report.json: images with no predictions at confidence 0.25; detection counts at 0.25 and 0.50; predictions per class and photo; possible overlapping contradictory classes.
- overlays/*.jpg: annotated pictures to inspect actual failures.
- predictions.json: reproducible, hashed model-output receipt.

Unlabeled prediction counts **cannot measure accuracy**.

## Optional: complete human-reviewed object annotations

For each photo, annotate every visible item from ALL seven target classes using YOLO format: class_id, normalized x_center, y_center, width, height. Use an empty label file only when no target objects are visible after manual inspection. Use unique photo stems.

Run the same command using a new output folder, with --labels /path/to/label-folder --labels-reviewed. Then report.json also contains per-class object precision, recall and F1 at both confidence thresholds. This is still not disposal-bin accuracy.

The completeness flag is an attestation; a script cannot establish that the human labeled every target object.

## Diagnostic questions

1. **Threshold effect:** Are objects present at confidence 0.25 but absent at 0.50? Inspect confidence calibration before retraining.
2. **True misses:** If objects are missing at both cutoffs, investigate school-domain shift, size, clutter, lighting, label completeness and source class definitions.
3. **Cross-class conflicts:** Check large-IoU boxes with contradictory classes, especially plate versus hinged container.
4. **Missing training labels:** A source dataset focused on plates might show an unlabeled bottle in the same image. Standard seven-class YOLO training can treat it as background.
5. **Sorting correctness:** School acceptance, material and container emptiness need separate adjudication.

## Repair path for one unified YOLO11 model

- Audit the complete original→unified seven-class ID mapping and maintain an image-level annotation coverage ledger. Do not concatenate partially labeled YOLO exports blindly.
- Fully annotate a representative set of mixed, crowded cafeteria images, including genuinely empty negative images. Use model-assisted labels only as drafts; human reviewers must check the entire scene.
- Split by source and school capture session, not by random image; keep independent future school sessions untouched until the final evaluation. Already inspected school photos become development examples.
- Train a new pretrained YOLO11n or YOLO11s candidate on audited data, keeping the four-specialist baseline frozen.
- Select on development validation only. Compare recall per class, mAP50–95, missed-object frequency and real measured latency. Evaluate the final locked configuration on the new independent school test.

No accuracy improvement or new trained model is established by adding the diagnostic tool.
