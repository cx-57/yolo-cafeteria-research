# Waste Detection v3 — revised triple audit (October 10, 2026)

Source: Roboflow `charles-xie/waste-detection-v3-tunsu-ti5cb`, v1 export (ZIP README says v2), CC BY 4.0.

**Change:** NHS uses the white disposable bowl and kraft soup tub shown in the source. These belong in new object class **10 = bowl**. No previous IDs change. Per user's request, no temporal thinning.

## Check 1 — Integrity

- Original: **923 JPGs** (877 train, 46 validation) with 11 source categories.
- Revised output: **331 image/YOLO label pairs**, **342 object boxes**, one current class per photo.
- Original polygons converted to axis-aligned detection boxes; **10 small edge corrections**.
- All 331 images decode successfully, have valid 5-value YOLO boxes and matching paired labels, and contain no byte-identical or pixel-identical duplicates. Raw source images were not changed.

## Check 2 — NHS suitability

| Current class | Images | Objects |
|---|---:|---:|
| 1 plate | 14 | 14 |
| 3 utensil | 38 | 40 |
| 5 napkin | 52 | 54 |
| 8 wrapper | 105 | 105 |
| 9 bottle | 47 | 47 |
| 10 bowl | 75 | 82 |
| **Total** | **331** | **342** |

The original **250** eligible frames are all retained, including **199** previously held as similar consecutive video frames. Added **42** matching white/kraft bowls from source category `Compost-Plate-Bowl`, **33** disposable white soup bowls from `Trash-Bowl-Cup`, plus **6** additional compatible white paper plates. **592 other frames excluded** (including glass/reusable bowls, off-target cups, cans, foil, retail packaging, transition frames and uncertain examples). White food bowls and kraft tubs are accepted, **not all source objects labeled bowl/cup**.

## Check 3 — Duplication and leakage

- No exact image-byte or pixel duplicates among the 331.
- Source remains heavily correlated; many kept photos are adjacent frames, explicitly requested by the user.
- The original Roboflow train/valid split shares recording groups; at least one group overlaps within the retained 331 images. **All 331 are training-only candidates**.
- Reserve separately captured real NHS cafeteria photos for validation/final testing. Do not report accuracy on mixed frames as independent generalization.

## Files and limitations

The curated ZIP `Waste_Detection_v3_NHS_Untrimmed_331.zip` contains original JPEGs, remapped current 11-class bounding-box labels, the per-photo provenance `manifest.csv` and this audit. The revised `Waste_Detection_v3_923_Image_Audit_Revised.csv` records all 923 decisions and their original statuses; these deliverables are downloadable in the chat and **are not yet committed as image binaries to this GitHub repo**.

A three-pass screen includes contact-sheet inspection of all source frames, higher-resolution review of bowls/ambiguous classes, structural/geometry checks and duplicate/group checks. It is **not** proof that every visible target object was exhaustively reannotated; complete multiclass review is still needed before a defensible scientific benchmark.
