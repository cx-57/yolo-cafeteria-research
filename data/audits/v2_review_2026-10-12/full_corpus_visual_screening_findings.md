# Visual annotation screening — all 3,096 photos (2026-10-12/13)

## Scope and verification standard

**Completed:** Visually inspected all **84 existing annotated contact sheets**, containing each of the **3,096 processed images** (the contact sheet source counts sum to 3,096). Each thumbnail shows the source YOLO bounding boxes and class designation. See [all_3096_visual_screening_register.csv](all_3096_visual_screening_register.csv) for one row per exact repository image, including its inspected contact sheet.

**Not completed:** Full-resolution inspection of *every individual object annotation* in every photo, including verifying every small instance, geometrically exact box, and missing target class. Contact sheets are approximately 40 images per sheet and **cannot** prove those things. In the register, `full_11_class_verified_by_this_screening = no` on **all 3,096 rows**, intentionally. The pre-existing 16 Codex reviewed overrides and 242 food-only LVIS review records remain separately distinguished; in this pass five directly viewed full-resolution food images were specifically examined, two received approved replacement labels.

The original V1 images, split, and original labels remain untouched. Corrections below affect **V2-only copied annotations** when built with `--use-codex-review`.

## Definite fixes included in this pass

| Image and size | Before | After in V2 reviewed overrides |
|---|---|---|
| `data/processed/food_american/images/train__download_jpg.rf.69af084c6f657aae6cb53fb448c69dfd.jpg` (225×225) | Three visibly separate chicken drumsticks, one central food box | Retain center food box; add one for each clearly visible left/right drumstick (3 total) |
| `data/processed/food_american/images/train__images_jpg.rf.27e5ce52d5650e34e6dfff7f3452c2fa.jpg` (260×194) | Two distinct burgers merged in one food box | Replace with one food box per burger (2 total) |

Each replacement includes the **entire corrected set** of labels and a SHA-256 of the original V1 train label. The dataset builder refuses it if the original source changes. The reviewed overrides file now has **18 images** (the existing 16 plus these two).

## Additional precise issues, flagged but NOT silently corrected

| ID | Source image (under `data/processed/food_american/images/`) | Visual evidence | Next action |
|---|---|---|---|
| 1353 | `test__images_jpg.rf.2e1375c3d05007414011f60f20dd3797.jpg` | Burger has food box, fries all around it and visible plate are not labeled | Full-resolution portion/plate policy review and boxes for fries + visible plate |
| 1360 | `test__images_jpg.rf.8a509a19412cb10f262e8d5d0aa2adf6.jpg` | Several pieces of fried chicken overlap, but only one central food box | Establish separate-piece versus whole-serving annotation policy, then relabel |
| 1622 | `train__images_jpg.rf.c84c19777d1d686a2a03373684b31d9d.jpg` | Burger is labeled; fries to its right and a white dinner plate are not | Review/add food and plate boxes as separate visible target objects |

Do not treat `photo visually inspected` as synonymous with `every target class exhaustively labeled`. These unresolved images are recorded as follow-up, **not approved V2 fixes**.

## Source-level visual screening (thumbnail evidence)

| Source | Images | Findings |
|---|---:|---|
| condiments | 44 | Condiment sachet boxes generally trace packets; almost identical isolated tabletop setting |
| dry_wet_waste | 233 | Bottles and wrappers appear but the source is outdoor litter, not a cafeteria. Some class-5 `napkin` candidates are torn printing/regular paper rather than napkins (five prior V2 corrections exist); more full-res review needed |
| plate (LVIS) | 344 | Occluded plates, stacks, very small portions and overlapping boxes need careful full-resolution geometry checks. Food on plates is often absent from original plate-only labels; Codex's 242 reviewed LVIS food overlays partly address this |
| tartanhacks | 62 | Isolated utensils and napkins, repeated camera positions, limited background diversity |
| trash_detection | 197 | Roadside wrappers and bottles; many boxes appear aligned but labels are not exhaustive for all visible classes and domain mismatch is large |
| utensil | 128 | Utensils generally correctly classed; groups/overlapping implements and packaging are not consistently exhaustively separated |
| waste_detection_v3 | 331 | Many video-like consecutive views of near-identical staged napkins, bottles, bowls and wrappers in one room; high scene correlation. Codex corrected select duplicate bowl/utensil labels, but a full per-frame inspection is pending |
| food_american | 380 | **Confirmed missing multiple food objects** and merged instances, corrected two images above. Stock/promotional images, some plates and secondary foods unboxed, inconsistent food-portion granularity |
| macworx | 121 | Hinged-container boxes generally cover objects; most photos show the same container on the same surface at different angles |
| school_lunch | 467 | Individual foods and cartons largely boxed. **Bowls, utensils, portions inside dishes, and other target-class objects are often unannotated**, because upstream annotations focus on food/cartons; do not claim complete 11-class coverage |
| school_lunch_asian | 130 | Same partial food/carton supervision and repeated cafeteria tray viewpoint; large bowls of soup/rice visible but bowls themselves often unboxed |
| tetra_pak | 659 | Carton boxes generally follow cartons; many photos are repeated rotations on the same conveyor/background and retail international packs that differ from NHS cartons |

**Total:** 3,096 images, 84 annotated thumbnail sheets examined.

## Evaluation implications

- A valid-looking box is not evidence that the *other visible target objects in the same image* were labeled. Ultralytics' default object detection loss can treat unlabeled true objects as background, especially when combining category-specific datasets.
- Existing 100-epoch V1 scores and newly labeled V2 scores are not directly interchangeable when validation annotations/split membership differ.
- Source image repetition should be grouped by original capture, not treated as independent training evidence. The V2 builder already repairs exact-name/byte-identical cross-split conflicts; other near duplicates need dedicated review.
- No independent NHS test set exists. It is not required for development, but partial 7-of-11 validation class support makes all-class mAP meaningless.

## Suggested continuation of exhaustive review

1. Inspect every photo at original resolution with bounding boxes and compare against the entire 11-class ontology, not merely its source dataset's categories. Review the `food_american` missed portions, LVIS plate occlusion and the 34 dry/wet `napkin` candidates first.
2. Log per-photo reviewer, checked class set, coordinates/precision, and `complete=yes` only when all visible target objects are accounted for. Prioritize multi-object scenes and varied real-world backgrounds.
3. Add V2 SHA-pinned, image-level replacement labels only for clear cases; avoid inventing boxes for uncertain partially visible objects or promoting unverified paper-boat drafts into evaluation.
4. Run `python3 scripts/v2.py build --provisional --use-codex-review --output data/training-v2-screened-2` with a **new** output directory, run tests, and evaluate on one frozen validation set.

**Outcome:** The entire corpus has now received an initial *visual screening*, which revealed systemic label incompleteness. It has **not** received 3,096 photo-level full annotation approvals, and no higher mAP is claimed.
