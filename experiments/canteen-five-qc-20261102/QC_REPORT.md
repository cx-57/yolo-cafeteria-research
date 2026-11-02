# Canteen Food Waste — five-image QC (experimental only)

Visual review: 5 source images individually inspected at stored pixel resolution; human-verified: 0. Nothing on main or frozen model results changed.

Class mapping: 0 food, 1 plate, 2 utensil, 3 hinged_container, 4 drink_carton, 5 wrapper, 6 bottle.
Decision: open serving bowls are class 1; cups/glasses/trays excluded. Edible meat attached to bone is provisionally food, bare bone not automatically food. One tangled transparent physical wrapper should get one box.

Individual counts:
- valid/12_img.jpeg: 13 (2048x1152), isolated rice grains remain review question.
- train/21_img.jpeg: 11 (1500x2000), spurious fork annotation overlapping spaghetti already absent in prior provisional labels; two genuine utensils retained.
- train/IMG_1428.jpeg: 7 (2268x4032), +1 plate (ceramic meat bowl at 1530,1270,2262,2050); one clear plastic wrapper revised to 1010,655,1860,1505. Meat on bones and film boundary remain provisional.
- valid/28_img.jpeg: 3 (792x1408), +1 plate (soup bowl at 0,80,791,1105). Soup and partial spoon retained.
- valid/9.jpeg: 3 (1536x2040), translucent material at right edge cannot be confidently identified as wrapper; flagged.

Counts: food 19, plate 6, utensil 11, wrapper 1; others 0. TOTAL 37. Two boxes added, one edited. All labels five-field YOLO, class 0..6, positive dimensions and coordinates within 0..1.

QC blockers: consistent micro-fragment/rice grouping rule; translucent film boundary; meat/bone disposal policy; unidentified transparent object in valid/9; possible source-split near-duplicate leakage. Retain Roboflow train/valid as source metadata only, not an approved leakage-safe split.

The original five JPEG byte streams and annotated PNG previews were exported in the local QC ZIP, NOT included in this branch. No GitHub Actions artifact was generated. Command Center must review ZIP. STOP after five; do not train or expand.
