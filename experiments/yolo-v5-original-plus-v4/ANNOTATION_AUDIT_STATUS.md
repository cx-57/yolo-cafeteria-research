# V5 original-data annotation audit — ONGOING, 2026-11-04

**Do not call the original 1,572 photographs fully annotated or visually checked.**

## Worker B flagged-187 QC — selective merge, 28 reclassifications withheld (2026-11-04)

**Latest provisional V5 artifact:** `YOLO_V5_CommandCenter_B187_QC_selected_merged1572.zip`, built against `YOLO_V5_CommandCenter_C39_QC_merged1572.zip`.

Worker B delivered `Worker_B_QC_187_Flagged_Reviewed(1).zip`, with 187 assigned 640×640 photographs, final YOLO labels, 187 overlays, 47 worker second-pass sheets, per-image before/after hashes, and an 80-row unresolved-case list.

**Verified before integration:** All 187 assignments match the authoritative Worker B ledger; source-image SHA256s, old-label SHA256s, revised-label SHA256s, normalized geometry, and overlay presence all passed. Worker B reported 73 label files changed, 114 unchanged, with 35 box tightenings, 10 deletions and 28 nominal `wrapper→carton` reclassifications.

**Critical visual finding:** All 28 proposed class-4 replacements describe highly blurred green/purple packaging repeatedly visible in the background of conveyor photographs. Magnified source crops do *not* establish whether it is a paper drink carton, flexible wrapper, or unrelated packaging. Worker B itself marked those cases unresolved. Some old red wrapper boxes also had poor geometry. **None of the 28 class-4 reclassifications was accepted**: the merged archive leaves these 28 prior label files unchanged, flagged for manual adjudication. They should be **temporarily excluded from training** because their retained class-5 labels may also be wrong; a 28-image exclusion CSV is delivered in this conversation. Repeated, near-identical conveyor frames are not independent evidence of an object's identity.

**Accepted 45 label revisions:** 35 existing class-4 carton boxes were tightened to exclude opaque time-stamp overlays / off-image edges (as demonstrated in visual comparison sheets), and 10 unsupported wrapper boxes on/crossing background/rails were removed. The other 114 label files did not change. All 187 worker images are still AI-provisional; independent human verification of unchanged cases was not performed.

**Updated counts:** V5 remains **2,093 train + 97 val + 71 test = 2,261** photos. Training instances are now food **3,753**, plate **2,085**, utensil **1,314**, hinged container **121**, drink carton **1,208**, wrapper **607** (minus 10), bottle **211**. No original image bytes changed. All ZIP CRC tests passed; every one of 2,261 manifest image-and-label SHA256s checked; **all 336 val/test image/label files remain byte-identical**.

**Remaining AI worker flags:** A 29, B **80** (down from 187 as reported by Worker B but not human resolved), C 33 → **142 total**. Rechecking and human adjudication are still required; notably the 28 held changes should not enter a fully trusted training corpus. The original V4 cross-class completeness audit and a truly seven-class held-out test remain outstanding.

Outputs:
- `YOLO_V5_CommandCenter_B187_QC_selected_merged1572.zip` — full merged candidate (no accepted class-4 additions from blurry backgrounds)
- `V5_WorkerB_187_QC_CommandCenter_Selected_Patch.zip` — 45 accepted labels and decision log
- `V5_CC_WorkerB_187_QC/WorkerB_187_Selected_Decisions.csv` — per-file accept/withhold provenance
- `V5_CC_WorkerB_187_QC/All_Workers_142_Unresolved.csv` — remaining flagged queue
- `V5_CC_WorkerB_187_QC/WorkerB_28_Images_To_Exclude_Until_Manual_QC.csv` — conservative training exclusion recommendation.

**No model training occurred.** No fully verified seven-class mAP should be reported from this corpus.

---

## Worker C flagged-39 final QC integrated (2026-11-04)

**Latest provisional V5 artifact:** `YOLO_V5_CommandCenter_C39_QC_merged1572.zip` (2,093 train / 97 val / 71 reserved test). Starting archive: `YOLO_V5_CommandCenter_A_Final50_QC_merged1572.zip`.

Worker C submitted `Worker_C_39_Flagged_Images_Final_QC(2).zip`, with **39** previously flagged original training photographs, original images, before-and-after labels, overlays, per-image hash and correction log, and an unresolved-cases CSV. Two uploaded copies of this QC submission were byte-identical.

- **11 of the 39 label files changed; 28 unchanged.** The edits consist of **7 added target boxes, 8 removed boxes, 0 class reassignments, 12 bounding-box adjustments**. All 11 changed photos and final overlays were independently visually inspected by the Command Center; all changes were accepted as **provisional**, not human-adjudicated ground truth.
- Examples: removed a spurious bottle label from distant linear debris (C234), deleted two false bottle boxes around aluminum cans/unknown cylindrical litter (C362), tightened identifiable plastic bottles (C306/C346), added two small plate and two food boxes in a café scene (C384), split distinguishable food items and tightened a bottle in a refrigerator scene (C403), and removed food-wrapper labels applied to generic transparent litter or a non-food rigid blister pack (C406/C424). A carton mockup (C158) was tightened but is flagged as a domain-shift sample.
- **Worker C unresolved: 33**, down from 39. Six cases are AI-resolved, **not independently human-confirmed**. Worker A unresolved: 29; Worker B unresolved: 187. **All-worker outstanding QC: 249 unique images**.
- Source-assignment checks passed for all 39 C sequence IDs and original images/labels: worker original image SHA256s and baseline label SHA256s exactly matched the *current* pre-merge V5 archive. All YOLO class IDs and normalized box geometry validated, including intentionally empty label files for photos without a confident target.
- Whole merged archive verified via **ZIP CRC, all 2,261 image/label manifest SHA256s and per-image annotation counts**. All **97 val + 71 reserved test** images and label files (336 total files) remain **byte-for-byte unchanged**. This QC submission concerns original train photos even when names contain `__valid__` because they were sourced from a separate dataset's original validation subset.
- Updated V5 training box counts: food **3,753**, plate **2,085**, utensil **1,314**, hinged_container **121**, drink_carton **1,208**, wrapper **617**, bottle **211**.
- Updated master assignment ledger, original review-status ledger, merged manifest, merge report, worker-specific QC log and `all_workers_249_remaining_uncertainties.csv` are inside the merged ZIP. Standalone outputs: `Worker_C_Final39_CommandCenter_QC/CommandCenter_C39_FinalQC_Integration_Report.json`, `Worker_C_Final39_CommandCenter_QC/Worker_C_33_Remaining_Uncertainties.csv`, `Worker_C_Final39_CommandCenter_QC/CommandCenter_ALL_249_Remaining_QC.csv`.

**Scientific limitations:** no independent full human audit; 249 original photos remain flagged; original V4 cross-class annotation audit and a seven-class held-out benchmark are still missing. **No model training occurred.** Binary archives delivered in the conversation, not committed to GitHub.

---

## Worker A final 50-image uncertainty recheck integrated with one correction withheld (2026-11-04)

**Latest provisional V5 archive:** `YOLO_V5_CommandCenter_A_Final50_QC_merged1572.zip` (2,093 train, 97 validation, 71 test).

Submission: `V5_Worker_A_Final_50_Flagged_QC(1).zip` contains 50 previously flagged Worker A training photographs (15 from initial Macworx batch + 35 from the prior 86-image recheck), all 50 latest labels, current-before-review labels, final overlays, original images, QC pairs, per-image audit, hashes, and an unresolved-29 CSV.

- Worker reports **50 individually inspected**, **2 label files modified, 48 unchanged**, **21 marked confidently resolved by the worker**, **29 still uncertain**.
- Command Center verified all 50 assignment IDs, exact image bytes and SHA256, existing labels matching the *current* V5 archive, new label SHA256, seven-class normalized box coordinates (with floating-point tolerance), every submitted image and final overlay, audit counts and ambiguous-case rows.
- **Accepted Macworx sequence 115:** extend class-3 hinged-container bottom bounding edge to encompass faint visible lower base. Source photo independently compared before/after; adjustment looks justified.
- **Withheld Utensil sequence 288:** worker proposed class-5 transparent wrapper across a large mixed utensil arrangement. Independent source/photo inspection could not distinguish a genuine plastic sleeve boundary, and the proposed box encompasses separate packaging/empty space; **this label was not merged**. Its uncertainty remains open for manual adjudication. The latest archive preserves the previously merged label for that image.
- **Final action:** only **one training label file** changed; all seven class box counts unchanged. The worker's 21 claimed resolutions remain AI-provisional, not human-verified ground truth.
- **Outstanding flags:** A **29**, B **187**, C **39**, totaling **255 distinct original training photographs**. B/C counts reflect earlier worker flags that have not yet been independently resolved. 1,572/1,572 original images have had an AI annotation pass, but this does not mean complete high-confidence labels.
- Archive was rebuilt without altering source images. **ZIP CRC passed**, **all 2,261 image and label SHA256 manifest records passed**, and **all 336 validation/test image + label files were byte-for-byte identical** to the prior archive. 2,093 training + 97 validation + 71 test photos. No model training performed.
- Deliverables in the current chat: latest V5 archive, `V5_CC_WorkerA_Final50_Integration_QA.json`, `V5_CC_WorkerA_Final50_29_Remaining_Uncertainties.csv`, and `V5_CC_AllWorkers_255_Residual_QC_Queue.csv`. Binary archives are not stored in Git.

**Important:** Worker A final50 and other AI corrections must not be represented as independently human-reviewed annotations. The merged V4 images still require cross-class annotation auditing, and V5 validation/test currently lack three of the seven classes.

---

## Worker A flagged-86 second QC integrated (2026-11-04)

**New provisional V5 archive:** `YOLO_V5_CommandCenter_Aflag86_QC_merged1572.zip`, derived from `YOLO_V5_CommandCenter_ABC_merged1572_provisional.zip`.

Worker A submitted `V5_Worker_A_Flagged_86_QC_Reviewed(1).zip`, with all **86** originally flagged photos and current labels, final overlays, per-image hashes, audit, and uncertainty report. The worker marked **51 AI-resolved** and **35 still uncertain**. **3 image labels changed** and **83 remained identical** to the currently merged dataset.

Corrections accepted after Command Center individually viewed all **three** revised before/after image pairs:
- School Lunch A146: tighten class-5 wrapper around upright food packet, excluding empty tray.
- Utensil A284: add one class-5 bounding box to a transparent cutlery sleeve; exact overlap/utensil identities remain uncertain.
- Utensil A400: extend existing class-2 broken-fork box to include a detached tine fragment.

All 86 assigned Worker A sequence/image IDs, original source SHA256s, old-label/current-archive SHA256s, revised SHA256s, class IDs, normalized box geometry, per-image counts, and overlay-file presence/decodability passed. Source pixels unchanged. Entire merged archive reverified: **2,261 image/label SHA256 pairs**, 2,093 training / 97 validation / 71 test, ZIP CRC pass, byte-for-byte unchanged validation/test (336 files). **New train box totals:** food 3,748; plate 2,083; utensil 1,314; hinged_container 121; drink_carton 1,208; wrapper **620** (+1); bottle 216.

**Outstanding uncertainties:** Worker A's *earlier first 100* Macworx batch has **15 additional unadjudicated** cases not covered by this 86-image handoff, so Worker A remains **50 outstanding** (15+35); B has 187; C has 39; **276 outstanding across the three workers**, assuming no overlapping IDs. This is an AI-level flag-resolution tally, **not independent human validation**, and unflagged items can still be imperfect. V4 cross-class review and seven-class held-out labels remain outstanding; no model training occurred.

Delivered alongside the full archive: `V5_CC_WorkerA_86_QC_Integration_Report.json`, `V5_CC_WorkerA_86_Review_Log.csv`, `V5_CC_WorkerA_50_Remaining_Uncertainties.csv`.

---

## Worker C final 325 merged — all original V1-era images provisionally AI-reviewed (2026-11-04)

**Command Center milestone: 1,572 / 1,572 original V1-era training photographs processed by three annotation workers. None remains unassigned or pending an initial AI worker pass. This does NOT mean independently human-verified or complete seven-class ground truth.**

Input `Worker_C_Images_101-425_Reviewed(1).zip` contains **325** assigned images, labels and final overlays (131 Tetra Pak and 194 Trash Detection), audit/status CSV, original/revised label hashes and uncertainty cases. The worker's original source metadata redundantly includes sequence 100, but the actual submitted top-level `assignment.csv` and `audit.csv` correctly cover **101–425 exclusively**, with 325 unique images.

- Worker C reported **51 additions, 15 deletions, 3 reclassifications, 146 bounding box corrections**; **176 label files changed, 149 unchanged**. Relative net object changes: **+6 food, +1 utensil, +3 drink cartons, +24 wrappers, +2 bottles**, other classes unchanged.
- **39/325 C photos explicitly flagged** for ambiguous background litter, packaging type, cropped bottles, etc. Some flag notes caution against classifying arbitrary non-food packaging as `wrapper`.
- The review correctly **removed a spurious class-4 drink-carton annotation from an unrelated cardboard cat house** (C117). Selected high-risk visual reviews generally supported revised labels; no claim that all images were independently reexamined.
- Command Center verified **325/325 assignment IDs**, original image SHA256, old/current training label SHA256, revised worker label SHA256, seven-class normalized coordinate geometry, all 325 final-overlay image formats, and duplicate-label checks. **49 selected original/overlay comparisons** were visually inspected, prioritizing high-risk large changes and flagged cases. This was not an independent full 325-image human QC pass.
- **Merged output:** `YOLO_V5_CommandCenter_ABC_merged1572_provisional.zip`. The archive contains **2,093 train, 97 val, 71 test** images. ZIP CRC and **all 2,261 image/label SHA256 manifest records** verified; **336 validation/test source files byte-identical** to the previous Worker-B-complete archive.
- Current V5 training object counts: **food 3,748; plate 2,083; utensil 1,314; hinged_container 121; drink_carton 1,208; wrapper 619; bottle 216.**
- All 1,252 Command Center worker assignments are marked provisionally AI-reviewed (A 402, B 425, C 425); the initial 320 originals had earlier AI review. **At least 327 distinct worker-assignment images are flagged for further review**: Worker A first batch 15, Worker A remaining batches 86, Worker B 187, Worker C 39. Flagging is imperfect; unflagged cases are not guaranteed correct.
- Deliverables from this conversation include the full merged archive, `V5_CC_C325_QC/Worker_C_Final325_Integration_QA_Report.json`, `V5_CC_C325_QC/Worker_C_39_Uncertainties.csv`, and `V5_CommandCenter_Master_Assignments_AllAIReviewed.csv`. These binary files are **not** committed to GitHub.

**Before claiming seven-class evaluation or publication-quality annotations:** independently adjudicate flagged cases, audit cross-class completeness in the 521 new V4 training images (and its validation/test), and construct a seven-class holdout. The V5 validation/test currently lack labels for three of the seven classes; do not report misleading seven-class mAP. No model training occurred during this merge.

---

## Worker B remaining 325 images integrated (2026-11-04)

**Overall Command Center: 1,247 / 1,572 original images provisionally AI-reviewed; 325 still pending (Worker C sequences 101–425). Worker B is complete (all 425 assigned photographs).**

Input: `Worker_B_Reviewed_Images_101_to_425(1).zip` contains 325 uniquely assigned Tetra Pak training images (640×640), 325 revised YOLO labels, 325 final overlays, assignment list, per-image audit, hashes and ambiguity log.

**Independent Command Center QC:** Verified all 325 exact sequence IDs against master assignment ledger, all 325 original image hashes, exact pre-merge label hashes, all corrected label hashes and normalized box coordinates, classes and class-count consistency, ZIP CRC and overlay image decodability. Checked same-class duplicate box overlaps and independently visually spot-checked **39 selected source/overlay pairs**, prioritizing large geometry changes and added classes. High-risk previews looked broadly plausible; tiny background package objects remain ambiguous and should not be presumed correctly classified without human QC.

- **325/325 label files revised.** Net additions: **+10 drink_carton, +38 wrapper, +4 food, +2 utensil, +1 plate**. No box removals reported; source photographs unchanged.
- **168/325 images are explicitly flagged uncertain by Worker B** (69 in seq 101–200; 13 in 201–300; 63 in 301–400; 23 in 401–425). In particular, 38 background-package wrapper labels and cropped carton edges require human adjudication before claiming exhaustive reliable annotations.
- Candidate dataset: `YOLO_V5_CommandCenter_Bcomplete_merged1247.zip`. All **2,261 image and label SHA256 manifest records** independently reverified; ZIP CRC passed. All **97 validation + 71 reserved test** image AND label files (336 total files) remain byte-identical to prior `YOLO_V5_CommandCenter_Acomplete_merged922.zip`. Same split: 2,093 train, 97 validation, 71 test.
- Seven-class training box totals: food **3,742**, plate **2,083**, utensil **1,313**, hinged_container **121**, drink_carton **1,205**, wrapper **595**, bottle **214**.
- Updated `merge_report.json`'s previously stale `annotation_audit_progress` and `annotation_audit_status` to 1247/1572, and updated the review-status CSV / master assignment ledger in the archive to match. Worker C remaining: 325 images.
- Additional output files: `V5_CommandCenter_Master_Assignments_After_Bcomplete.csv`, `V5_CC_B325_QC/Worker_B_168_Uncertain_Images.csv`, and `V5_CC_B325_QC/Worker_B_Remaining325_Integration_QA_Report.json`. Large archives are delivered in the chat, not checked into GitHub.

**Scientific caveat:** This is a provisional worker annotation merge with selective AI visual review; it is not an independent full-resolution gold-standard inspection of all 325 images. The 168 flagged cases, prior A/B uncertainty flags, incomplete V4 cross-class annotations, and absence of a seven-class validation benchmark still prevent claims of full seven-class annotation quality. No model training was performed.

---

## Worker A remaining 302 images integrated (2026-11-04)

**Overall Command Center: 922 / 1,572 original images provisionally AI-reviewed; 650 remain (Worker B: 325; Worker C: 325). Worker A all 402 assigned images submitted.**

Input `V5_Worker_A_Images_101_to_402_Reviewed(1).zip` contains all 302 assigned Worker A photos (sequences 101–402), revised labels, old labels, full-size final overlays, three batch logs, and 102 source+overlay QC pairs for the final batch. Breakdown: 18 Macworx, 159 School Lunch, 125 Utensil.

**Independent Command Center verification:**
- All 302 image SHA256s match the prior V5 archive; all 302 original label hashes match exact pre-merge originals.
- Every assignment path/order, old/new log hash, existing class ID, full normalized box boundary, image dimension and final overlay encoding passed. No same-class duplicate box pair with IoU >0.80 was detected.
- 192 of the 302 label files changed; 110 remained unchanged (mostly utensil-only images).
- Class-box net deltas: **food −3; plate +525; utensil +438; carton +1; wrapper +148; hinged container 0; bottle 0**.
- Visual spot-audit of **33 selected original and final overlay pairs** across all three sources did not confirm a definite error. This is **not independent review of all 302 images**.
- **86 of 302 images flagged uncertain by Worker A**, including 3 Macworx, 61 School Lunch, 22 Utensil. Batch 003 review logs say overlay generated, not an independently human-adjudicated full-res review. Batch 004 explicitly says native-pixel review sheets plus selected standalone images, not standalone inspection of every photo.
- No automatic reclassification or unsourced annotations were added during Command Center integration.

**Merged candidate:** `YOLO_V5_CommandCenter_Acomplete_merged922.zip`. Validated ZIP CRC, **all 2,261 image/label SHA256s**, matched label line counts to manifest, and byte-for-byte unchanged **97 val / 71 test** images and labels. Totals remain **2,093 train / 97 val / 71 test**. Full train box totals now: **food 3,738; plate 2,082; utensil 1,311; hinged container 121; drink carton 1,195; wrapper 557; bottle 214**. No training occurred.

Outputs in conversation: updated full V5 archive, `V5_CommandCenter_A302_Integration_QA.json`, `V5_CommandCenter_Master_Assignments_After_Acomplete.csv`, `V5_CommandCenter_A302_uncertain_images.csv`.

**Next:** Complete B 101–425 (325 photos) and C 101–425 (325 photos). Resolve flagged images and audit V4 cross-class omissions before any scientific claim of complete seven-class annotations.

---

## Worker B + Worker C Batch001 merged — 200 images (2026-11-04)

**Command Center overall: 620 / 1,572 originals marked AI-reviewed; 952 remain.**

Worker B (`Worker_B_Batch_001_100_Reviewed(1).zip`): first 100 Tetra Pak photos, 640×640. **93 corrected** and **7 unchanged** label files. Carton boxes **163 → 178**, plus **1 food, 3 plate, 1 utensil**. In total **20 new boxes**. **19 images flagged** for uncertain crop/boundary/secondary-object interpretation. Seven larger box-geometry changes were spot-checked visually.

Worker C (`Worker_C_Batch_01_100_Reviewed(1).zip`): first 100 Tetra Pak photos, 640×640. **100 revised existing cartons**, one target per image, with **no additions, removals or reclassifications**. No worker-provided uncertainty flags. One larger geometry change spot-checked visually.

**Integration method:** Checked each worker's exact assignments (sequences 1–100 only), matched source images and original label SHA256 values against the already-merged `YOLO_V5_CommandCenter_A001_merged420.zip`, verified all 200 revised label SHA256 values and box geometry, checked overlays exist and decode, and checked for same-class duplicate labels with IoU > 0.75. Independently viewed **35 selected original-to-overlay comparisons** (24 B, 11 C), emphasizing added objects and unusually large geometry changes. No definite error found in that sample; this was **not exhaustive independent visual review**.

**New candidate archive:** `YOLO_V5_CommandCenter_ABC001_merged620.zip` (2,093 train / 97 val / 71 test); updates 193 revised label files, manifest hashes and object counts, Command Center assignments, original review status, and QA report. **All 2,261 image and label SHA256 records passed**; ZIP CRC passed; validation and test image/label bytes were compared against the previous archive and remained unchanged. The existing model checkpoints were not modified; no training.

**V5 training counts after merging B+C:** food 3,741; plate 1,557; utensil 873; hinged_container 121; drink_carton 1,194; wrapper 409; bottle 214.

**Next assignments:** A 101–200, B 101–200, C 101–200. Worker A has 302 remaining, Worker B 325, Worker C 325. The 19 flagged B cases and any previously flagged A cases remain provisional and should be human-reviewed before gold-standard claims. V4 cross-class exhaustiveness is not established.

---

## Worker A Batch001 — Command Center integration (2026-11-04)

**Worker A submission:** 100 Macworx original training images (assignment A sequences 1–100). The worker reported opening all source photos at 640×640 and inspecting each final overlay. The independent Command Center audit verified assignment order, unchanged source image SHA256s, old-label SHA256s against the pre-worker V5 archive, final YOLO coordinate bounds, and full review log consistency. It **visually spot-checked 30 before/after source images, including all nine with large box discrepancies**; this is not independent full human verification.

- **95 existing class-3 `hinged_container` boxes refined; five unchanged.** No added objects, deletions, reclassifications, or other target classes in the 100 Macworx photos.
- **15 edge/transparent-boundary cases retained as uncertain for independent human QC.** The nine largest changes visually looked plausible in the audit sample; the maximum reduction was to about 48% of the original box area.
- **`YOLO_V5_CommandCenter_A001_merged420.zip`** created as a *separate* complete V5 candidate archive. Only these 100 train labels plus audit metadata/status/manifest hashes changed relative to `YOLO_V5_partial_annotation_audit320.zip`.
- Verified 2,261 image/label records (2,093 train / 97 val / 71 test), all image and label SHA256 checks, all normalized box geometry checks, archive CRC, and **byte-for-byte unchanged validation/test files**.
- **Current overall review count: 420 / 1,572; 1,152 pending.** Worker A still has 302 assignment photos; Worker B has 425 and Worker C has 425. Updated assignment ledger: `V5_Command_Center_Master_Assignment_After_A001.csv`; original-image status: `V5_Command_Center_Review_Status_After_A001.csv`.
- **No training was performed** and no frozen V1/V4 checkpoints were touched. All corrections are **AI-provisional**, not human-adjudicated ground truth.

Deliverables are attached in the Command Center conversation, not committed as 200+ MB binary blobs to GitHub.

---

## Command Center — remaining 1,252 original images assigned to three workers (2026-11-04)

Assignments derived from `YOLO_V5_partial_annotation_audit320.zip` and the latest review-status CSV. **All are previously PENDING training originals.** Each assignment contains original photos, their existing provisional label files, per-file manifest hashes, and a source-ordered checklist. There are **zero overlapping image IDs across workers** and **no validation/test files in the worker archives**.

| Worker | Dataset assignment | Count | Worker archive |
| --- | --- | ---: | --- |
| A | 159 School Lunch + 118 Macworx + 125 Utensil | **402** | `V5_Worker_A_402_Pending_Images.zip` |
| B | First 425 remaining Tetra Pak images by filename | **425** | `V5_Worker_B_425_Pending_Images.zip` |
| C | Remaining 231 Tetra Pak + 194 Trash Detection | **425** | `V5_Worker_C_425_Pending_Images.zip` |

The worker archives and `V5_Command_Center_Control_Files.zip` (master assignment CSV, three copy-paste prompts, checksums, and prior review ledger) are delivered as attachments in the Command Center conversation. Do not falsely mark any assigned photos complete until each worker delivers image-by-image corrections and its audit log.

Handoff: 100-photo batches per worker (last remainder shorter). Every batch must return corrected labels **including unchanged labels**, a per-image audit ledger, old/new SHA256 hashes, second-pass visual overlays, and ambiguous-case flags. Integrate patches only by canonical label paths with hash verification; freeze validation/test, retain provenance, and resolve ontology differences consistently across workers. Remaining V4 cross-class annotations must be audited separately before claiming exhaustive seven-class training data.

---

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
