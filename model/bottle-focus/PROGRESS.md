# Bottle specialist experiments — 2026-10-23

Target: at least **85% bottle AP50**. Best freshly verified result at this
snapshot: **84.0784%**, using a saved single-class YOLO11s checkpoint at
640 pixels with ordinary CPU inference, batch 4 and NMS IoU 0.5. The target has not been reached.

The fixed development validation set has **246 images and 67 bottle boxes**.
All validation images, bottle annotations and splits are preserved.
Source fingerprint:
`6668d2b7f0b16243640c14efdbb6bdffbb7f83fc880459befd58e8afcbfa2296`.
Validation protocol:
`29423591896198d41f5bd16392b51f06c7c3d7254f8a903b3dda74569cfe7de7`.
These are development results, with no independent test claim.

| Saved model / experiment | Ordinary AP50 | Augmented AP50 |
| --- | ---: | ---: |
| Pretrained YOLO11x, bottle output only | 40.21% | 45.65% |
| YOLO11s head fine-tune, best of eight epochs | 67.11% | 69.08% |
| YOLO11x head fine-tune, two completed epochs | 60.28% | 60.23% |
| YOLO11s context resampling, best of eight epochs | 68.39% | 71.33% |
| YOLO11s context resampling, best of 23 epochs | 73.67% | 71.22% |
| YOLO11s gentle feature adaptation, best of eight epochs | 82.08% | 82.14% |
| Same checkpoint, NMS IoU 0.5 | — | 82.65% |
| YOLO11s head-only 35-photo supplement, best of six epochs | 82.99% | 80.75% |
| Half/half head average, TTA NMS IoU 0.5 | 83.02% | 83.35% |
| Same head average, ordinary NMS IoU 0.5 | 83.68% | — |
| Head interpolation alpha 0.75, ordinary NMS IoU 0.5 | 83.73% | — |
| Quarter quality-head blend, ordinary NMS IoU 0.5 | 84.07% | 80.68% |
| Context-balanced head, best saved model of 12 epochs, NMS IoU 0.5 | 83.14% | — |
| Half context/quality head average, CPU ordinary NMS IoU 0.5 | **84.08%** | — |

All scores are independently reloaded evaluations. The non-averaged
head-only checkpoint’s training-time validation score was 83.05%; the saved-model score above takes precedence.
Training validation uses a different batch size and saved checkpoints are
serialized at half precision, so the two scores need not match exactly.

The verified incumbent is retained locally at
`verified-incumbent/bottle.pt`; its settings and SHA-256 are recorded in
[`verified-incumbent/verification.json`](verified-incumbent/verification.json).
SHA-256: `2cd1efe2d7c187df05ed223ba2a18e147393358179f223c264663a2114812289`.
Existing Git ignores exclude model weight files; reports and experiment code
are versioned, while checkpoints remain local.

Completed comparisons:

- Unfreezing later features at learning rate 0.0001 did not beat the
  initialization in three epochs.
- Inference at 960 and 1280 pixels did not beat 640-pixel augmented inference.
- A training-only BatchNorm recalibration pass changed no learned parameters,
  but reduced validation AP50 for both tested checkpoints.
- Training-only resampling repeats 124 existing bottle-positive context
  photos four times per epoch. Every other training photo remains included
  once. There are 1,056 unique training images, 1,428 exposures and 890 bottle
  exposures per epoch; no new photos were admitted.

Context resampling completed 23 epochs. Its epoch-16 averaged candidate
reached 72.73% augmented AP50. Gentle feature adaptation then started from
the verified 73.67% ordinary-inference checkpoint, retaining fixed BatchNorm
statistics and reducing the learning rate to 0.00001. It completed 13 epochs; epochs 9–13 did not improve on epoch 7.
At 832 pixels, ordinary/augmented AP50 were 74.72%/80.16%; 640 pixels
with tighter NMS reached 82.65%.

The new `external-data-v7` experiment adds 19 freshly downloaded LVIS
photos with 36 plastic beverage bottle boxes to training only. The
AI-assisted visual reviews are explicitly not human verification.
Validation remains fixed at 246 photos / 67 boxes. A separate 548-photo
Mendeley source was sampled and rejected as supplied after 24 photos
showed repeated scenes and incomplete/out-of-bounds annotations.
The 19-photo supplement trial completed six epochs without beating its
starting ordinary AP50; the original checkpoint was retained. A combined
35-photo / 65-box supplement is now being tested with the feature layers
frozen and only the detection head adapting at learning rate 0.00003.
The combined supplement trial completed six epochs. Its epoch-3 checkpoint
reached 82.99% in independently reloaded ordinary inference. Default
multi-scale TTA reached 80.75% (NMS 0.7) / 80.63% (NMS 0.5), and a
full-resolution horizontal-flip variant reached 81.00%; these did not improve
on ordinary inference. A single half/half head average reached 83.02% ordinary / 83.35% augmented
AP50. Corresponding frozen backbone weights and all BatchNorm running
statistics were identical before averaging; only head parameters differed.
The head-only run completed epochs 7–8 without beating epoch 3. The
uncompleted epoch 9 was interrupted for a training-label audit; optimizer
state through epoch 8 is preserved. All 754 bottle-negative training photos
are being screened for possible missing targets before further fine-tuning.

The latest `hard-negative-polish-v11` trial uses 41 external photos with
80 bottle boxes, five reviewed training-photo holds and 12 repeated hard
negatives. A training-only scan found one definite missing-bottle-label photo
and four uncertain photos; no original data or validation labels were changed.
Mosaic is disabled and scale jitter is 0.25. Six epochs completed without
improving the ordinary control; the best newly trained native score was
81.02%. The original saved checkpoint is retained. The
`gentle-feature-polish-v12` trial unfroze layers 10 onward at learning
rate 0.000003 on the exact same training copy. Its native AP50 declined
80.22% / 76.92% / 76.12% across three completed epochs, so epoch four was
interrupted; optimizer state through epoch three and the control are retained.

The `expanded-context-v13` trial adds 16 further reviewed LVIS photos with
29 boxes (57 external photos / 109 boxes total), retains the five quality
holds, and removes extra hard-negative repeats. It uses 1,108 unique photos /
1,651 exposures and the previously successful head settings: freeze 23,
AdamW 0.00003, mosaic 0.5, scale 0.5, fixed BatchNorm running statistics.
Six epochs completed, best native 81.46%; independently reloaded augmented
AP50 was 78.35%. It did not improve the incumbent. V11 and V12 newly
trained candidates also scored 82.29% and 80.81% with augmented inference
and NMS 0.5. All validation bytes and labels remain unchanged.

The full 124-photo original context-positive cohort was reviewed with
targeted crops. Ten photos / fourteen boxes were conservatively held for
uncertain material/type; these are not ten confirmed label errors. The
`scope-clean-feature-v14` trial uses the five prior negative holds plus those
ten context holds, the same 57 external photos / 109 boxes, 1,098 unique
photos / 1,611 exposures, and 451 unique training bottle boxes. It starts
from the pinned pre-soup V6 checkpoint (edab…); the 83.68% global incumbent
remains protected. Later features adapt at 0.00001 with mosaic 0.5, scale
0.5 and fixed BatchNorm statistics. Native AP50 fell 76.67% / 71.36%
in two completed epochs, so epoch three was interrupted and the control
retained.

A fixed four-view rotation TTA test (0/90/180/270 degrees, full resolution,
NMS 0.5) reached 76.81% and was rejected. Both independent geometry/model
integrity tests and all 235 project tests passed; 246 adapter calls were
verified. Two neighboring head interpolation points, alpha 0.25 and 0.75,
are being evaluated as ordinary single saved models under the same settings
as the incumbent alpha 0.5. Alpha 0.25 scored 82.98%; alpha 0.75 scored
83.73%. Four-view rotation Weighted Boxes Fusion scored 83.12%;
two-view original/horizontal-flip fusion scored 80.89%. Both were rejected;
geometry/fusion tests and the full 235-test project suite passed.

The fixed quality-head blend uses 0.75 of the alpha-0.75 V8 head average
and 0.25 of the V11 quality-trained head. Frozen backbone tensors and all
BatchNorm statistics were identical. Freshly reloaded ordinary inference
reached 84.0711%; default augmented inference reached 80.6760%. The exact
ordinary checkpoint is retained. This winner combines V8 and V11 lineage;
the V11 branch used 41 external photos / 80 boxes and five negative-image
holds. Later 57-photo and 15-hold copies are separate trials.

The active `context-balance-v20` trial retains the scope-screened 1,098
unique-photo / 451-box pool, repeats 114 original context photos eight
times, and includes every other photo once. Its 57 external photos each
appear once; there are 1,896 exposures. It uses seed 43, pinned edab
initialization and the frozen-feature head recipe at 0.00003. The verified
84.07% global incumbent is protected. A half/half quality-head blend is
checked on CPU with its device explicit; it scored 83.45% and was rejected.
The context-balanced run completed 12 epochs; epoch 3 remained best. Its saved model reached 83.1407% ordinary AP50 at NMS IoU 0.5 and 82.6453% augmented AP50 at NMS IoU 0.7. Phase-12 checkpoints and reports are preserved.

Regression-only and classification-only branch blends scored 83.7310% and 83.8528%; both were rejected. A half parameter average of the 84.0711% quality model and the context-balanced epoch-3 head reached **84.0784%** on a fresh CPU evaluation. The frozen backbone and BatchNorm statistics were identical. This normal single-model checkpoint is the current incumbent. Default NMS IoU 0.7 reduced its AP50 to 83.3168%.

The larger `medium-head-v24` trial has now started from official pretrained YOLO11m weights with 20,053,779 parameters after single-class projection. A real non-square input confirmed unchanged decoded boxes and the original COCO bottle scores. It uses the exact V20 reviewed data and fixed validation, head-only training, fixed BatchNorm statistics, learning rate 0.0001, 640 pixels and batch 2. The six-epoch head phase finished; epoch 3 remained best. Fresh ordinary/augmented checks reached 68.8293%/72.7199%. The 84.0784% smaller-model incumbent is protected.

Run using the existing local runtime:

```sh
PYTHONPATH=/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/py312:$PWD/scripts BOTTLE_PHASE_EPOCHS=6 .venv/bin/python -u model/bottle-focus/context-balance-v20/run.py
```

The cached dataset is prepared by
`validation-now/prepare_data_snapshot.py`. Individual experiment directories
contain their scripts, configurations, metrics, hashes and evaluation output.

## Paper-informed continuation

The original higher-scoring papers and their training/data differences are
recorded in [PAPER_REVIEW_20261023.md](PAPER_REVIEW_20261023.md). Numerically
higher external scores are not head-to-head results on our fixed set.
The small incumbent scored 83.6731% with ordinary NMS 0.6 on MPS, so that
setting was rejected. A matched ordinary MPS batch-one, NMS 0.5 control
scored 83.4980%. Official SAHI 0.12.8 sliced inference at original resolution
(384-pixel slices, 25% overlap, full-image prediction, NMS 0.5) scored
76.3749% and was rejected. All 246 source images and 67 boxes were retained;
740 slices and 246 additional full-image predictions were traced. Two
independent coordinate tests, including the actual YOLO loader, passed.

The V25 medium-model feature adaptation has started from the immutable
V24 phase-six best snapshot, SHA e01fc63b…8934. It unfreezes from layer 10,
uses AdamW 0.00001, fixed BatchNorm statistics, training batch 1 with nbs 16,
640 pixels, and up to 100 total epochs with mosaic disabled in the last ten.
Six-epoch review phases preserve optimizer state. Results remain pending;
this is a paper-informed experiment, not an exact paper replication.

A fourth LVIS staging/review batch yielded 40 usable photos. Initial AI
review marked 20 eligible; second review held three more for ambiguous
identity/partial targets or incomplete box coverage. **17 photos / 20 boxes**
remain eligible candidates. None has entered V24 or V25 training. The
validation-overlap heuristic flagged another already-held candidate. All
reviews are AI assisted and are not human verification.
