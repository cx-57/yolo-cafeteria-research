# Waste Detection v3 — three-check image audit (October 10, 2026)

Source: `charles-xie/waste-detection-v3-tunsu-ti5cb`, Roboflow export v1 (ZIP README title: v2), CC BY 4.0.

All **923 images** were included in annotated contact sheets; particularly ambiguous plates, utensils and wrappers were examined at larger size. This is a frame-by-frame contact-sheet visual screen, **not** a guarantee that all annotations are exhaustively correct at full resolution.

## Check 1: Structural

- **877 train, 46 valid, 0 test** images; original taxonomy of **11 categories**, **789 boxes**, **159 empty-label transition/background frames**.
- All photos decoded; 0 exact duplicate image hashes.
- **25 original bounding-box rows** exceeded strict normalized image boundaries. The 51-image final package required **one** minor edge clipping; every final image/label pair now passes normalized detection-box and class-ID checks.

## Check 2: School-object suitability

| Original source class | Labeled images | School-object decision |
|---|---:|---|
| Compost-Cup | 63 | Reject (not a current model class) |
| Compost-Lid | 50 | Reject |
| Compost-Napkin | 52 | 52 visually eligible brown paper napkin images |
| Compost-Plate-Bowl | 87 | 8 plausible plates; 79 bowls/containers rejected |
| Compost-Utencil | 50 | 38 plausible disposable utensils; reject unsuitable/blurred/mixed cases |
| Recycle-Aluminum-Can | 100 | Reject; no can class |
| Recycle-Aluminum-Foil | 52 | Reject; no foil class |
| Recycle-Food-Packaging | 55 | Reject retail noodle boxes |
| Recycle-Plastic-Bottle | 47 | 47 eligible beverage bottles |
| Trash-Bowl-Cup | 89 | Reject |
| Trash-Wrapper | 120 | 105 eligible snack wrappers; reject foil lids/ambiguous mixed examples |

One frame has two source-class annotations; the source class totals overlap once.

## Check 3: Diversity and split leakage

- **Four source video groups** occur across both Roboflow train and valid partitions; original validation is not independent.
- **250** visually eligible images, but **199** held due to temporally adjacent frames (thinning: ≥6 source-frame indices apart within a video/class).
- **51** curated training candidates remain: 2 plates (model ID 1), 9 utensils (ID 3), 7 napkins (ID 5), 24 wrappers (ID 8), 9 bottles (ID 9).
- **673** images rejected overall (409 out-of-taxonomy, 159 empty/uninformative, 79 bowl mismatches, 11 utensil mismatches, 14 unsuitable foil lid/packaging, 1 ambiguous mixed scene).
- Final 51 images: matching pairs, zero exact duplicates and zero dHash pairs at Hamming distance ≤3.

**Interpretation:** The ZIP contributes potential *training data only*. They are all taken from indoor demonstration videos, not distinct natural NHS cafeteria sessions. New independent NHS scenes remain necessary for final validation/testing. Captures have not been imported into GitHub yet; add them via the documented ZIP workflow when ready. The separate per-image CSV audit supplied with the dataset records all 923 decisions.
