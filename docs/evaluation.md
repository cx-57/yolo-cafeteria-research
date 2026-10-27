# Evaluation

The active configuration is pinned by checkpoint SHA256. Five detectors run on
the same 124 development images under one ordinary 640-pixel inference method;
no ensemble, verifier or test-time augmentation. AP uses Ultralytics matching
at IoU 0.50:0.05:0.95 and its standard integration. The 313 objects span eight classes.

| Class | AP50 | AP50–95 | Objects |
|---|---:|---:|---:|
| Food |96.60%|64.64%|81|
| Plate |76.40%|62.00%|49|
| Utensil |81.03%|54.30%|65|
| Hinged container |98.88%|58.47%|12|
| Drink carton |89.32%|64.57%|29|
| Wrapper |89.56%|67.53%|21|
| Bottle |87.21%|74.34%|11|
| Condiment packet |60.94%|41.96%|45|
| Macro mean |84.99%|60.98%|313|

Labels are provisional and AI-assisted. The cohort excludes 49 known cross-model
training/background overlaps, holds 13 incompletely annotated candidates and
includes 22 fully reviewed packet-positive images. Exact bytes, decoded pixels
and available capture/group identities were checked. Photographer/session
independence and absence of COCO pretraining overlap are not certified.
Historical validation was repeatedly used for selection. Small class counts
and mixed public-source scenes limit interpretation. An independent cafeteria
test with human-certified labels is still required.

The long packet run used 204 fitting and 66 group-separated monitor images from
reviewed training data, fresh COCO-pretrained YOLO11s, a 40-epoch cap and early stopping.
Its final shared result was 78.64% AP50 / 56.09% AP50–95; packet 10.12% / 2.89%.
It was rejected. The current five checkpoints remain selected. This outcome
is retained in external `artifacts/packet-focus/big-run-20261027/`.

Full historical manifests, prediction receipts, reviews and method notes remain
in the external research store's `data/` and `artifacts/` directories. The
maintained evaluator rejects incomplete class coverage, missing/changed images,
incompatible frozen receipts and malformed detections. Output is never silently
overwritten. Reused predictions can recompute metrics, but are not fresh inference.

Layout references: [Ultralytics](https://github.com/ultralytics/ultralytics) and
[YOLOv5](https://github.com/ultralytics/yolov5). The source repository holds code,
configuration, tests and concise results; large local assets live separately.
