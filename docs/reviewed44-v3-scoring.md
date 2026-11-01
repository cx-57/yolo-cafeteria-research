# 44-photo manually reviewed development evaluation — V3 versus original

**Current status:** 44 source images and **104 AI-visual draft bounding boxes** are packaged in a standalone image review ZIP delivered in the conversation. Drafts are NOT human-verified ground truth. These same 44 public photographs were repeatedly inspected when building V2/V3/V4, so resulting scores are **development diagnostics**, not an independent test.

This cohort contains 12 photos originally sourced as hinged-container candidates, 15 utensils, 9 bottles, 6 cartons, and 2 wrappers. **Those are source hints, not verified image labels.** The drafts include additional visible food, plate, and utensil boxes. Some boxes and class assignments may be incomplete or incorrect.

### Label review (required for defensible mAP)

1. Download the ZIP named `cafeteria-44-test-annotation-review.zip` from the conversation, unzip it on your Mac, and open `index.html`.
2. For each photo, **inspect all seven class types**: `food, plate, utensil, hinged_container, drink_carton, wrapper, bottle`. Remove wrong boxes; move poorly placed boxes; draw any missing boxes, including classes not indicated by the original source hint. Individually confirm each box; mark the photo reviewed. There may be many objects.
3. Photo 11 is a visually uncertain hinge-versus-nonhinged tray; carefully check the actual hinge. Photos 34, 35 and 36 have dense/partially occluded bottles and were **preexcluded from the earlier provisional scoring review**. The evaluator defaults to scoring the **other 41** photos; do not enable all 44 unless every visible bottle in those photos has been labeled and verified.
4. Click **Export** to download `cafeteria-review-export.json`. This is your reviewed object-label receipt. Image IDs and SHA-256 values must match the original ZIP.
5. Run the code below. The program rejects unreviewed photos, unconfirmed AI boxes, mismatched images, invalid boxes or unavailable classes. It then evaluates BOTH frozen original and V3 against exactly the same labels and images.

```bash
cd ~/Desktop/yolo-cafeteria-research
git pull --ff-only origin main

./scripts/run_specialists.sh evaluate \
  "$HOME/Desktop/yolo-cafeteria-research-assets/artifacts/test-label-audit-20261031/audited-all248.json" \
  --config "$HOME/Desktop/yolo-cafeteria-research-assets/artifacts/hinged-candidate-v3-retry/candidate-specialists.json" \
  --output "$HOME/Desktop/yolo-cafeteria-research-assets/artifacts/hinged-candidate-v3-retry/test-248-v3.json" \
  --device mps
```

**The above is the separate 248-image historical test, not the 44-image score.** For the **reviewed 44-source-photo evaluation**, run:

```bash
cd ~/Desktop/yolo-cafeteria-research
git pull --ff-only origin main
./scripts/run_specialists.sh review44 \
  --bundle "$HOME/Downloads/cafeteria-44-test-annotation-review.zip" \
  --review "$HOME/Downloads/cafeteria-review-export.json" \
  --v3-config "$HOME/Desktop/yolo-cafeteria-research-assets/artifacts/hinged-candidate-v3-retry/candidate-specialists.json" \
  --output "$HOME/Desktop/yolo-cafeteria-research-assets/artifacts/hinged-candidate-v3-retry/reviewed44-metrics.json" \
  --device mps
```

For a genuinely **44-image** score rather than the default **41-image eligible subset**, thoroughly review all three previously excluded dense bottle photos, then append `--include-excluded` to the command.

This produces original and V3 overall seven-class mAP50 and mAP50–95, per-class AP and side-by-side deltas. It does **not** retrain or change either model. Do not reuse reviewed 44-image outcomes for further model selection and then report this as a fresh independent test. A sealed, new school-captured cohort is needed for that claim.
