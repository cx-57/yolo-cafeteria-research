# Bottle image candidates — 2026-10-16

## 2026-10-17 additional round

The paired optimizer probes did not improve bottle AP50. The preserved epoch3 checkpoint remains the AP50 comparison model. A new positive training supplement was reviewed from the official local LVIS train archive using the existing streaming reader:40 links attempted,33 downloaded,8 admitted with28 plastic beverage bottle boxes;25 downloaded candidates remain unapproved. Download failures are preserved in `additional_lvis_downloads.json` and `additional_lvis_single_downloads.json`. The538-ID index is a candidate pool, not an approved dataset.

`approved_training_reviews_context.json` is the hash-bound new sign-off. `mutual_validation_duplicate_audit.json` documents bidirectional mutual feature screening of all20 admitted external photos against226 validation images, with a positive resized-JPEG control. No duplicate was flagged; the screen is not proof of full capture independence.

The new dataset is `../bottle-context-reviewed`:1302 images (1076train/226val), train650plate/399wrapper/378bottle and validation237/89/45. All1294 prior photos, label bytes and splits are preserved. It has not yet been trained or evaluated. The new recipe and user-run command are at `model/focus-three-class-improvement/EXPANDED_EXPERIMENT.md`. Earlier counts and commands below describe the2026-10-16 snapshot.


Update:12 fully reviewed photos with20 plastic beverage bottle objects have been added to the isolated `../bottle-enriched-clean` dataset. All1282 prior images/labels/splits and the226-image validation set were preserved. `approved_training_reviews_clean.json` is the authoritative image-bound sign-off. Candidate screen statuses below remain the original download-screen snapshot. Ambiguous cases are not implicitly approved.

Downloaded **138 images** from146 attempted links into `images/`. **54 candidates** passed the preliminary existing-image similarity screen:22 LVIS water-bottle scenes,19 TACO litter scenes,13 river scenes.84 downloaded images are held as possible duplicates;8 links failed. Similarity holds are conservative and can contain false positives. This is not a claim that54 are fully verified, independent training examples.

**21 images are prioritized for full review**, emphasizing small plastic drink bottles, multiple bottles, occlusion, and indoor/social contexts. The first two `priority-preview-*.jpg` files show that shortlist. `review_queue.csv` provides paths and flags; `candidates.json` retains original URLs, labels, image hashes, dimensions, source group hints, and nearest existing-image matches. `summary.json` contains exact screening totals.

## Sources and suitability

- [LVIS](https://github.com/lvis-dataset/lvis-api): used existing official local LVISv1 validation annotation archive, category1162 water_bottle, and original Flickr links. Meal tables, meeting rooms, kitchens, picnic and sports scenes complement isolated-bottle training images. A water_bottle label does not establish plastic material; reusable metal and glass examples require exclusion. Source split names are provenance only. These COCO photographs are not suitable as a new independent test of a COCO-pretrained model.
- [TACO](https://github.com/pedropro/TACO): selected official category5 Clear plastic bottle, prioritizing source boxes below4% image area and multiple-bottle scenes. Existing `trash_detection` photos overlap this source substantially, so downloading more TACO images does not automatically add new examples. Official annotations were saved locally. Image-level license fields are absent from the source JSON; image rights/attribution require verification from original source links rather than assuming the code's MIT license covers photographs.
- [Plastic Bottles in Rivers](https://github.com/m0-n/Plastic-Bottles-Dataset): one photograph per each of34 source sessions was sampled, with Pascal XML labels retained. The author declares the data public domain. Water/river backgrounds are a domain mismatch for a cafeteria; keep these secondary and test their contribution separately. The repository's headline5592 refers to bottles and is inconsistent with some README image-count statements; the fetched tree contains1001 JPG files, not5592 verified independent images.

[Labeled Waste in the Wild](https://web.fe.up.pt/~jsc/publications/conferences/2019JoaoSousaWVC.pdf) is a closer cafeteria source: its authors report1002 food-tray images and266 plastic-bottle labels, but the paper says access is by request. No author was contacted and no inaccessible images are counted here. SALSA was also inspected; its broad Plastic class needs new bottle-specific annotation and was not downloaded as a bottle dataset.

## Before training

The12 signed-off photos are training-only additions; both existing validation sets remain unchanged. Original11-class data and model files were not modified.

All downloaded images decode, and saved SHA256 hashes were checked. The screen compares against all3096 V2 source images using file hashes, decoded RGB hashes, and64-bit difference hashes across quarter-turn rotations (hold threshold10). It is heuristic; visually review resized/cropped duplicates and shared capture sessions before assigning any split. Source batches/session strings are provisional group hints, not proof of independence.

Verify plastic beverage identity and annotate **every** plate, wrapper and qualifying bottle in each image; do not import bottle-only labels as complete three-class labels. Keep one box per plate stack. Resolve source orientation/coordinate alignment before converting boxes.13 downloads initially differed from source annotation dimensions;10 still differ after EXIF orientation. Among the54 preliminary candidates, dimension issues remain in: river_008, river_009, river_012, lvis_000000015148, lvis_000000223182, lvis_000000394627. These are unresolved, not approved labels. Do not silently stretch or rotate labels to make a count pass.

The first gentle probe is invalid because Ultralytics rewrote a Samsung-trailer JPEG. The original downloaded file is retained; a separate cleaned derivative removes67 trailing metadata bytes with identical decoded pixels. The rebuilt clean dataset is starting a fixed80-epoch standard-augmentation run from COCO-pretrained YOLO11n. No accuracy improvement is assumed from downloading or adding photographs. Other candidates still require complete three-class review, coordinate correction where justified, and image/capture leakage checks.
