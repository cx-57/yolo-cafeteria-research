# Published-recipe small-model control

Test the actual full-fine-tuning recipe recorded in the WaDaBa authors' pinned
YOLO11m notebook (author-notebook.json). Its effective optimizer was AdamW
0.000909, despite a requested 0.01: auto ignored that requested learning rate.
The saved output shows all features trainable, 20 epochs, physical batch 16,
nbs 64, 640 pixels, mosaic 1.0 and close_mosaic 10. It logs 3,200 training
and 800 validation photos; it does not demonstrate an 11,000-image YOLO run.

Local control uses YOLO11s and physical batch 4 for Mac memory/compute, with
nbs 64 accumulation. Keep native training-mode BatchNorm and freeze no
features; assert live trainability at each epoch. Explicit AdamW 0.000909
prevents auto selecting a different rate for our single class. Use fresh COCO
bottle-projected initialization b28f0550, not a blended specialist checkpoint.
Retain the exact V20 reviewed training data and fixed 246-photo/67-box
validation. New round-four candidates are not admitted. Existing 84.0784%
weights stay protected. This is a hardware-adapted controlled recipe test,
not an exact reproduction or a cross-dataset performance comparison.

Initialization passed and training has started after preserving V25 epoch 2.
The fresh baseline is 26.9615% AP50. Live checks verify 9,428,163 trainable
parameters and 81 training-mode BatchNorm modules. The complete planned run is 20 epochs, with optimizer/state and best
weights saved each epoch and fresh saved-model checks afterward. A native
85% crossing triggers early saved-model verification.

Completed all 20 epochs. Best native epoch was 18 (64.3836% AP50).
Fresh saved-model ordinary640/batch4/NMS0.7 scored 64.5471%; native TTA
batch1/NMS0.7 scored 62.7979%. Selected weight SHA aabf5f18…c3b8.
Both are below the protected 84.0784% incumbent. Phase-20 snapshots retain
the selected weights and reports; optimizer state remains in last_resume.pt.
This hardware-adapted full-model recipe did not outperform our specialist.
It does not isolate learning rate from initialization, BatchNorm microbatch
or architecture differences and is not evidence that full fine-tuning
cannot help under other settings.
