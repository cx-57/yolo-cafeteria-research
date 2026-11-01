# Exploratory run on 44 provisional images (not an independent test)

This procedure **does not require human review** to generate predictions and browse visual overlays. It **does not** estimate mAP, accuracy, correct-bin eligibility, or independent field performance because no independent human object/bin truth exists. All candidate photos were previously source-screened and remain out of domain; no training, checkpoint selection or release change is authorized.

## Run on the Mac with the four frozen checkpoint files

Download the conversation artifact **cafeteria-44-image-review-kit-final.zip** into your Mac's Downloads folder. This contains images, source metadata, an optional annotation editor, and an unverified per-image manifest—not checkpoints.

From a terminal:

\`\`\`bash
cd ~/Desktop/yolo-cafeteria-research
git pull --ff-only origin main

./scripts/run_specialists.sh explore \
  "$HOME/Downloads/cafeteria-44-image-review-kit-final.zip" \
  --device mps \
  --out "$HOME/Desktop/yolo-cafeteria-research-assets/artifacts/exploratory-44-nov01"
\`\`\`

The output folder must not exist. The launcher uses the existing project's Mac Python environment. All four specialist checkpoints must already be under \`weights/\` and match the research freeze hashes. It refuses altered or missing checkpoint/config/policy files. This routine only operates on the **44 photo bundle** and not the separate 52 LeFood images.

Open the resulting \`.../artifacts/exploratory-44-nov01/index.html\` locally in a browser. The report directory contains:

- \`index.html\`: photo-by-photo annotated detector predictions with category labels and confidence.
- \`overlays/\`: prediction bounding boxes, not ground-truth annotations.
- \`detections.csv\`: one record per displayed detection with class, confidence and suggested outcome.
- \`summary.json\`: predicted class/disposal count distribution, images with no displayed detection, explicit null accuracy.
- \`predictions.json\`: full, pinned original inference receipt and sorting suggestions with per-image absolute file paths.
- \`source/\`: original extracted images, source metadata, and their verified SHA-256 hashes.

The runner rejects a changed ZIP member, repeated or missing images, corrupted or extra entries, changed frozen weights, and an existing destination folder. It checks frozen checkpoints before and after running. A source-category hint (e.g. \`bottle\`) is *not* a verified ground-truth label and is never converted to a score.

## Interpreting findings

This gives a concrete exploratory visual diagnostic without imposing a human review task. An empty image prediction may indicate a miss, but **cannot be counted as a false negative** without verified complete labels. An incorrect-looking box may be discussed qualitatively; don't invent false-positive denominators.

If you want actual independent recognition/compost/recycling accuracy in a paper, new sealed school images and verified object/bin annotations remain a requirement. Without them, write the results as development/prototype characterization only (248 previously exposed development test photos, 87.78% mAP50) and be explicit about incomplete independence and actual disposal correctness.

Do **not** retrain on these 44 provisional evaluation candidates; doing so would contaminate subsequent evaluation.

Run the tests for this utility with \`python3 -m unittest tests/test_explore_candidates.py -v\`. Six tests cover member/hash checks, accurate unverified output labels, and refusal to run altered model hashes. They do not prove inference quality.
