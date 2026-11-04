# V5 original-data annotation audit — ONGOING, 2026-11-04

**Do not call the original 1,572 photographs fully annotated or visually checked.**

## Batch02 — 50 more School Lunch photos completed (2026-11-04)

**Overall: 120 / 1,572 original V1-era photographs visually reviewed. 1,452 pending.** All 50 Batch02 source images were inspected at original 448×336 resolution, manually given bounding boxes for visible missing seven-class targets, and the 50 revised overlays individually QC'd.

- 193 NEW annotations: **107 plate/bowl**, **28 utensils**, **58 wrapper/packaging**.
- 3 originally food-labeled, completely obscured bread parcels were reclassified as **wrapper**.
- The post-annotation QC corrected multiple spoon/chopstick box locations and recovered missed chopsticks in images 36, 37, and 38.
- Original source image bytes unchanged; only these 50 train labels changed.
- Final model-data class counts in training: food **3,740**; plate **1,080**; utensil **576**; hinged container **121**; drink carton **1,176**; wrapper **242**; bottle **209**.
- Rebuilt V5 has 2,093 train, 97 validation, 71 reserved test images. Validation and test bytes and all manifest SHA256 hashes were verified unchanged; whole ZIP CRC test passed.
- `YOLO_V5_partial_annotation_audit120.zip` is the latest **provisional** dataset; `V5_School_Lunch_Batch02_50_Annotation_Patch.zip` holds original 50 images, revised labels, annotated overlays, audit logs and QC sheets; `V5_original_review_queue_after_batch02_50.csv` lists next pending images.
- Ambiguous cases remain marked: separate lidded food cups, freestanding metallic lids, snack packages. None have received independent human double verification. The V4 cross-class annotation audit is still outstanding.

Lesson for later batches: assess EVERY visible class, notably tiny packets and thin chopsticks. Review each utensil box at original resolution separately from the food-and-plate boxes; avoid mislabeling a concealed food parcel as visible food.

---

## Batch01 — 50 School Lunch photos completed (2026-11-04)

**At completion of Batch01: 70 / 1,572** original V1-era images individually visually reviewed at native resolution (50 new + 20 previously checked); **1,502 remained then; 1,452 remain after Batch02**. The entire 1,572-photo archive passed mechanical label scanning, but the 1,502 are **not** individually verified.

Batch01 selection: the first 50 *PENDING* School Lunch files in `original_review_status.csv`, in source order. Every selected source photo was inspected individually at 448×336 pixels and each revised overlay received a second visual QA pass.

Changes made to the 50 images:
- Added **106 plate/bowl boxes**, **57 wrapper/packet boxes**, **1 overlooked drink-carton box**.
- Corrected **two opaque wrapped snacks previously labeled as food** to class 5 (wrapper).
- **164 newly added boxes**, plus 2 reclassifications; no original V1/V4 model checkpoints modified.
- Source photo pixels remain identical; only 50 existing V5 training label files changed.
- All 50 label files passed class ID, normalized geometry, and per-image box-count checks. ZIP CRC verification succeeded, and all 2,261 manifest label SHA256 values match. Existing V4 validation (97) and test (71) files remain unchanged.

Seven-class V5 training box counts now: food 3,743; plate 973; utensil 548; hinged container 121; drink carton 1,176; wrapper 181; bottle 209.

**Remaining ambiguities for independent human QC:** single-serving food cups/lidded tubs (not currently counted as plates); an empty metallic lid; thin foil and plastic condiment packets. The 50-photo revisions are high-confidence *provisional* annotations, not an independently double-reviewed gold standard.

Deliverables generated in the chat:
- `YOLO_V5_partial_annotation_audit70.zip` — complete merged V5 dataset, includes both review passes and updated manifest/status ledgers.
- `V5_School_Lunch_Batch01_50_Annotation_Patch.zip` — 50 original photos, corrected labels, before-to-after review metadata, corrected overlays, and QA overview sheets.
- `V5_original_review_queue_after_batch01_50.csv` — global status ledger.

Next sequential batch starts at `school_lunch__010655000_jpg.rf.60946313e3a9640ce4bfff4f2d00c045.jpg`. Continue checking individual photos; do **not** assume all original School Lunch or other-source labels are complete.

---

## Prior 20-image audit (historical baseline)

The following notes describe the first 20-image pass, before Batch01 added 50 more images.

## Sources
- Original Day1: school_lunch (467), tetra_pak (659), macworx (121), utensil (128), trash_detection (197).
- All **1,572** original photographs were mechanically scanned for labels, box geometry, extreme dimensions and source-specific labeling risks.
- **20 of 1,572** were opened one-by-one at original pixel resolution (8 School Lunch; 3 each of Tetra Pak, Macworx, Utensil, Trash Detection).
- Candidate label revisions were made in 15 of the 20 inspected photos; five visually checked photographs retained original labels.
- **36 candidate additional boxes**: 18 plate, 10 utensil, 6 wrapper, 1 bottle, 1 carton. Six existing Tetra Pak/Macworx boxes and one bottle box were also tightened. Candidate boxes need independent second-person check, especially the ambiguous carton.
- **After this earlier pass, 1,552 original photographs were pending. Following Batch01, 1,502 remain pending.**

## Major observed issue

Original School Lunch labels only represent food and cartons, even when plates/bowls, cutlery, plastic bread wrappers and occasional bottles are plainly visible. The source-specific datasets are generally *not* exhaustive seven-class annotations. Reusing them unchanged as full seven-class negative labels can damage a unified detector.

The mechanically checked dataset also contains 396 edge-touching boxes and 26 very large boxes; these flags alone are not proof of error.

## Status of deliverables

A separate **partially audited V5 archive** retains the 2,261 image split, leaves validation/test unchanged, updates only the 15 corrected training label files, and keeps manifest label SHA256s synchronized. A compact patch ZIP contains individual corrections, full-resolution review evidence, audit status CSV and five source contact sheets. These large files are delivered in the conversation attachments, *not* committed to Git.

**Do not claim fully verified seven-class annotations: 1,502 original photographs and V4 cross-class completeness still require image-by-image review.**
