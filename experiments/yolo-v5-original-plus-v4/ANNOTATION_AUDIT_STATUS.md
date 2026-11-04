# V5 original-data annotation audit — ONGOING, 2026-11-04

**Do not call the original 1,572 photographs fully annotated or visually checked.**

## Batch04 — 100 further School Lunch originals reviewed (2026-11-04)

**Current progress: 320 / 1,572 original images visually reviewed; 1,252 pending.** The next 100 `PENDING` School Lunch images from the Batch03 review ledger were individually examined at native **448 × 336** pixel size, annotated with seven-class bounding boxes, and their full-resolution overlays reviewed in a second QC pass.

- Added **471 boxes**: **233 plate/bowl**, **146 utensil**, **85 wrapper**, **5 bottle** (capped plastic drinks), and **2 drink carton** (previously entirely unlabeled, images **24 and 88**).
- QC corrected carton omissions, tightened bread/noodle wrappers in **14, 34, 39**, adjusted chopsticks in **39 and 88**, and tightened image **100**'s seasoning-packet and spoon boxes. Image 88 needed its chopsticks moved **above** the spoon. Images **5, 25, 30, 34, 81** were treated as capped plastic bottles, not cartons. Ambiguous sealed food tubs were excluded from the seven target classes.
- Combined V5 **training box totals** after Batch04: food **3,740**; plate **1,554**; utensil **872**; hinged_container **121**; drink_carton **1,179**; wrapper **409**; bottle **214**.
- New dataset **`YOLO_V5_partial_annotation_audit320.zip`** (2,093 train, 97 val, 71 reserved test) includes 100 revised training labels, updated review ledger, merge report, per-image audit, and SHA256 manifest. Original source photos were not altered. Every image and label hash verified. The 97 validation and 71 test photos and labels were compared byte-for-byte against the prior `audit220` archive, identical. Full ZIP integrity CRC passed.
- **`V5_School_Lunch_Batch04_100_Annotation_Patch.zip`** holds originals, before/after labels, 100 final overlays, 25 QC sheets, image-level audit data, and the list of second-pass repairs. **`V5_original_review_queue_after_batch04_100.csv`** tracks all 1,572 originals with 1,252 remaining. Large archives delivered in the current chat, not committed to Git.
- **Limitations:** annotations were authored and visually checked by an AI, **not independently adjudicated by human annotators**. The V4 cross-class completeness audit is also not complete. These are provisional correction candidates and should not be called gold-standard ground truth or used for independent final claims.

### Lessons for next batch

1. **Audit source label omissions** even for a source's primary target class. Entire milk/drink cartons were previously missed in images 24 and 88.
2. Distinguish capped yogurt-style **drink bottles** from paper **cartons**; do not misclassify sealed pudding/yogurt tubs as a target object when the class ontology lacks cups.
3. **Individually verify chopstick and spoon bounds**, which sometimes overlap; geometric validity alone cannot prove correct placement.
4. Trim box extents on clear wrappers to the **visible bag**, not empty parts of the tray; food and wrapping are separately labeled only when identifiable.

---

## Batch03 — 100 more School Lunch photos reviewed (2026-11-04)

**Current total: 220 / 1,572 original V1-era photos visually checked; 1,352 pending.** All 100 pending School Lunch photos were inspected in the original **448×336** pixels, each given individual seven-class candidate corrections, and revised overlays visually checked.

- **474 added boxes:** 241 `plate` (includes bowls), 150 `utensil` (spoons/chopsticks), 82 `wrapper` (clear wrapping and condiment packets), 1 `drink_carton` (missing blue carton in source photo #33).
- **24 utensil coordinates corrected in second-pass QA**, after finding several chopstick boxes 20–40 pixels below their actual objects. The newly identified carton was also tightened to its visible extent after full-resolution source inspection.
- Existing labels and source image pixels are preserved except the 100 corresponding training label files; no frozen model checkpoints were changed. Original food boxes were preserved where food remained visible through wrapping. Ambiguous closed cups/lids outside the schema were flagged rather than forced into classes.
- New seven-class training box counts: food **3,740**, plate **1,321**, utensil **726**, hinged_container **121**, drink_carton **1,177**, wrapper **324**, bottle **209**.
- The revised full V5 dataset **`YOLO_V5_partial_annotation_audit220.zip`** still contains 2,093 train, 97 validation, and 71 reserved test images. Every image/label manifest SHA256 and every bounding box geometry check passed. Validation/test images and labels were independently compared byte-for-byte with the previous 120-review archive; full ZIP CRC passed.
- Batch patch **`V5_School_Lunch_Batch03_100_Annotation_Patch.zip`** includes the 100 original photos, old/new labels, final revised overlays, source SHA256s, per-image additions, flags, and audit metadata. Review queue **`V5_original_review_queue_after_batch03_100.csv`** tracks 1,352 pending. These large ZIP artifacts were delivered in the chat, *not* stored in Git.
- **Scientific caveat:** these are provisional AI visual annotations with a second AI coordinate/overlay QA pass, not independent human gold-standard annotations. The remaining 1,352 original images and V4's cross-class completeness issues have not been resolved.

### Lessons carried forward

1. Verify thin chopsticks against the unscaled 448×336 source or a line-localization check; contact-sheet coordinate transfer caused systematic vertical offsets in this batch, which were caught and corrected.
2. Original School Lunch source labels systematically omit plates/bowls, utensils and wrappers, and one clearly visible drink carton lacked its target label.
3. Keep `food` and `wrapper` as separate objects where edible food is visible through clear wrapping; avoid making up new classes for sealed cups outside the seven target classes.
4. Do not train or publish seven-class mAP from this still-incomplete corpus as if it were an independently audited detection benchmark.

---

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
