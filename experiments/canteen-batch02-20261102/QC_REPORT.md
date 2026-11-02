# Canteen Batch 02 — 10-image AI visual QC

Status: **PENDING COMMAND CENTER APPROVAL**. Source: Roboflow v1 COCO Segmentation, first ten images not inspected in prior provisional audit. Every image individually opened at its stored resolution. **Human verification: 0.** Files on this branch are experimental and must not be used as an approved training split.

Class mapping: 0 food; 1 plate (includes rigid serving bowls, provisional for glass serving dish); 2 utensil; 3 hinged_container; 4 drink_carton; 5 wrapper; 6 bottle.

53 boxes total: food 23, plate 13, utensil 16, wrapper 1; other classes 0.

Per-image before → after:
- train/69_img.jpeg: 8 → 2. Seven overlapping spaghetti/beef component annotations grouped into one coherent food mound.
- train/12.jpeg: 9 → 4. Overlapping pasta ingredients consolidated; missed fork added.
- train/3.jpg: 5 → 5. Missing plate added; potato and lettuce regions grouped coherently.
- train/30_03.jpg: 3 → 3. Sauce smear and isolated crumbs flagged as uncertain residues, not forced into huge mostly empty food boxes.
- train/IMG_1433.jpeg: 2 → 6. Large plate and cropped rigid bowl added; two meat-bearing bone clusters labeled provisionally, bare bones excluded.
- train/1.jpeg: 6 → 5. Erroneous second plate (bread/pizza) corrected to food and duplicate overlap resolved.
- train/13_img.jpeg: 9 → 7. Duplicate plate removed; coherent meat, pasta, salad portions corrected.
- train/66_img.jpeg: 4 → 6. Clear bread bag flagged as one wrapper; bread, grapes, stirrer added; drink mugs excluded from plate class.
- train/52_img.jpeg: 5 → 5. Bread, salad, salmon and chicken separated into coherent portions.
- train/17_img.jpeg: 6 → 10. Open glass baking dish, partial second serving dish, spoon, and top chicken included provisionally.

Source-image bytes, filtered COCO, every class/id/normalized numerical verification, annotated full-resolution previews, per-box audit, change log, SHA-256 manifest, near-duplicate provenance and four future 40-image batch assignments are in the **locally exported Batch 02 QC ZIP**, not on GitHub.

Unresolved high-priority cases: 30_03.jpg patchy sauce residue; IMG_1433.jpeg meat versus bare bone and edge-clipped foil-covered bowl; 66_img.jpeg clear bread bag and stirrer; 17_img.jpeg glass serving dish / partial-bowl policy. All labels remain AI-reviewed and subject to independent approval. No training, main modifications or Batch 03 work.

Remaining after Batch 02: 160 images; Batches 03–06 each planned as 40 original images (no expansion before approval).