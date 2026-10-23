# Larger pretrained bottle specialist

Training started; baseline and training results are pending. Use official YOLO11m weights
from the pinned upstream release URL in download.json. Project COCO bottle
class 39 into a single output, checking decoded boxes exactly and bottle
scores within float32 tolerance on a non-square real-model input.

Reuse the exact V20 training copy: 1,098 unique photos / 451 bottle boxes,
1,896 exposures, 114 original context photos repeated eight times, 57
external photos once each and 15 conservative quality holds. Fixed
validation remains 246 photos / 67 boxes; no independent test claim.

Head-only AdamW 0.0001, freeze 23, all BatchNorm running statistics fixed,
640 pixels, batch 2, nbs 16, seed 42, mosaic 0.5, scale 0.5 and six-epoch
phases. The smaller model's verified incumbent stays protected. A fresh
baseline is measured before training; no score claimed before evaluation.
