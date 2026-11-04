# V5 original-data annotation audit — PARTIAL, 2026-11-04

**Do not call the original 1,572 photographs fully annotated or visually checked.**

## Sources
- Original Day1: school_lunch (467), tetra_pak (659), macworx (121), utensil (128), trash_detection (197).
- All **1,572** original photographs were mechanically scanned for labels, box geometry, extreme dimensions and source-specific labeling risks.
- **20 of 1,572** were opened one-by-one at original pixel resolution (8 School Lunch; 3 each of Tetra Pak, Macworx, Utensil, Trash Detection).
- Candidate label revisions were made in 15 of the 20 inspected photos; five visually checked photographs retained original labels.
- **36 candidate additional boxes**: 18 plate, 10 utensil, 6 wrapper, 1 bottle, 1 carton. Six existing Tetra Pak/Macworx boxes and one bottle box were also tightened. Candidate boxes need independent second-person check, especially the ambiguous carton.
- **1,552 original photographs are pending full individual visual review.**

## Major observed issue

Original School Lunch labels only represent food and cartons, even when plates/bowls, cutlery, plastic bread wrappers and occasional bottles are plainly visible. The source-specific datasets are generally *not* exhaustive seven-class annotations. Reusing them unchanged as full seven-class negative labels can damage a unified detector.

The mechanically checked dataset also contains 396 edge-touching boxes and 26 very large boxes; these flags alone are not proof of error.

## Status of deliverables

A separate **partially audited V5 archive** retains the 2,261 image split, leaves validation/test unchanged, updates only the 15 corrected training label files, and keeps manifest label SHA256s synchronized. A compact patch ZIP contains individual corrections, full-resolution review evidence, audit status CSV and five source contact sheets. These large files are delivered in the conversation attachments, *not* committed to Git.

**Do not train V5 as an annotation-verified dataset until the remaining 1,552 original photographs and V4's additional cross-class annotation omissions are reviewed.**
