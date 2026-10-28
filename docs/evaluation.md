# Seven-class evaluation

Four frozen models run on the same 124 development images, with complete reviewed
coverage for the seven retained classes. Packet labels are excluded at the user's
request; no images or other class labels were removed. Class IDs 0–6 are unchanged.
Ordinary single-pass 640, NMS IoU 0.7, batch 1; no ensemble, verifier or TTA.
AP uses Ultralytics matching at IoU 0.50:0.05:0.95 and standard integration.

| Class | AP50 | AP50–95 | Objects |
|---|---:|---:|---:|
| Food |96.60%|64.64%|81|
| Plate |76.40%|62.00%|49|
| Utensil |81.03%|54.30%|65|
| Hinged container |98.88%|58.47%|12|
| Drink carton |89.32%|64.57%|29|
| Wrapper |89.56%|67.53%|21|
| Bottle |87.21%|74.34%|11|
| Macro mean |88.43%|63.69%|268|

The macro score differs from the old eight-class result because the task now
excludes condiment packet. It is not evidence that the retained models improved.

Labels remain provisional and AI-assisted. Known cross-model training overlaps
were excluded, and incomplete annotations were held. Exact bytes, decoded pixels
and available group/capture identities were checked. Capture-session independence
and absence of COCO pretraining overlap are not certified. Historical validation
was repeatedly used for selection, and some classes have few objects. An
independently collected cafeteria test with human-certified labels is still needed.

The seven-class manifest records its parent eight-class manifest hash. Full source
reviews, historical prediction receipts and packet experiments remain in the
external data/research store. Frozen predictions with an old eight-class recipe
cannot be silently reused as a new four-model receipt.

Layout references: [Ultralytics](https://github.com/ultralytics/ultralytics) and
[YOLOv5](https://github.com/ultralytics/yolov5).
