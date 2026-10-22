# Full-resolution head polishing with reviewed hard negatives

Training has started; no result is claimed yet. The starting saved detector is
the independently verified 83.35% AP50 half/half head average. Its ordinary
inference baseline is 83.02%; scores use the unchanged 246-photo / 67-box
bottle development benchmark. No independent NHS test result exists.

The training copy uses all 41 reviewed LVIS/TACO supplement photos with
80 bottle boxes, plus the original training pool with five image-level holds.
An audit of all 754 original bottle-negative training photos used both the
specialist and the original COCO YOLO11s detector. AI visual review covered
14 specialist proposals and 75 COCO proposals. Most were valid negatives:
glass bottles, jars, condiment containers, wrappers, cartons, cans and footwear.
One photo had unboxed plastic drinks bottles in a storefront window. That
photo and four material/scope-ambiguous photos are excluded from this copy;
original files and labels remain untouched. Twelve clear non-target photos
flagged by the specialist have four exposures instead of one. Decisions are
hash-bound in `negative-reviews.json`, are AI-assisted, and are not human
verification or an exhaustive audit of every training absence.

There are 1,092 unique training photos and 1,623 exposures per epoch. All
original bottle-positive training photos and boxes remain. Validation image
bytes, annotations and membership are unchanged. The original three-class
`verified_absence` field does not independently certify a bottle absence
after projection; the audit does not interpret its false value as a label error.

The detection head adapts at AdamW learning rate 0.00001; feature layers
0–22 and all pretrained BatchNorm running statistics remain fixed. Mosaic
is disabled and scale jitter is 0.25 to avoid repeatedly shrinking small
bottles in collages. The control and newly trained candidate are retained
separately, with six-epoch phases and optimizer continuation available.

Run with the existing local environment:

```sh
PYTHONPATH=/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/py312:$PWD/scripts BOTTLE_PHASE_EPOCHS=6 .venv/bin/python -u model/bottle-focus/hard-negative-polish-v11/run.py
```
