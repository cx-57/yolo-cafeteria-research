"""Fast fixed-grid inference diagnostic. No training or ground-truth-guided crops."""
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from torchvision.ops import batched_nms, nms
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import DetMetrics

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from crop_diagnostic import crop_rectangles
from inspect_predictions import parse_labels

NAMES = {0: 'plate', 1: 'wrapper', 2: 'bottle'}
DATA = ROOT / 'data/focus-three-class-improvement/reviewed-619'
OUT = Path(__file__).resolve().parent


def infer(model, picture, ox=0, oy=0, bounds=None):
    result = model.predict(picture, imgsz=640, conf=.001, iou=.7,
                           max_det=300, device='mps', rect=False, verbose=False)[0]
    detections = []
    for box, score, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
        if bounds:
            left, top, right, bottom, width, height = bounds
            # Suppress partial objects cut by internal tile edges. Real image edges remain valid.
            if ((left > 0 and box[0] <= 2) or (top > 0 and box[1] <= 2)
                    or (right < width and box[2] >= picture.width - 2)
                    or (bottom < height and box[3] >= picture.height - 2)):
                continue
        detections.append([box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy, score, int(cls)])
    return detections


def merge(rows, iou, agnostic=False):
    if not rows:
        return []
    values = torch.tensor(rows, dtype=torch.float32)
    if agnostic:
        keep = nms(values[:, :4], values[:, 4], iou)
    else:
        keep = batched_nms(values[:, :4], values[:, 4], values[:, 5], iou)
    return values[keep[:300]].tolist()


def update(metrics, validator, rows, truth, image_name):
    predictions = torch.tensor(rows, dtype=torch.float32).reshape(-1, 6)
    targets = torch.tensor([t[1] for t in truth], dtype=torch.float32).reshape(-1, 4)
    classes = torch.tensor([t[0] for t in truth], dtype=torch.float32)
    tp = validator._process_batch(dict(bboxes=predictions[:, :4], cls=predictions[:, 5]),
                                  dict(bboxes=targets, cls=classes))['tp']
    metrics.update_stats(dict(tp=tp, conf=predictions[:, 4].numpy(),
                              pred_cls=predictions[:, 5].numpy(), target_cls=classes.numpy(),
                              target_img=np.unique(classes.numpy()), im_name=image_name))


def main():
    started = time.monotonic()
    model = YOLO(str(ROOT / 'model/focus-three-class-improvement/stills-ft8/train/weights/best.pt'))
    assert model.names == NAMES
    validator = DetectionValidator(save_dir=OUT, args=dict(task='detect', mode='val'))
    modes = {name: DetMetrics(names=NAMES) for name in ['full_square640', 'tiles_iou05', 'tiles_iou07', 'tiles_agnostic05']}
    paths = sorted((DATA / 'images/val').glob('*'))
    assert len(paths) == 92
    for number, path in enumerate(paths, 1):
        with Image.open(path) as opened:
            picture = opened.convert('RGB')
            width, height = picture.size
            truth = parse_labels(DATA / 'labels/val' / path.with_suffix('.txt').name, width, height)
            full = infer(model, picture)
            combined = full[:]
            for left, top, right, bottom in crop_rectangles(width, height, .6):
                combined += infer(model, picture.crop((left, top, right, bottom)), left, top,
                                  (left, top, right, bottom, width, height))
            rows_by_mode = dict(full_square640=full, tiles_iou05=merge(combined, .5),
                                tiles_iou07=merge(combined, .7), tiles_agnostic05=merge(combined, .5, True))
            for name, rows in rows_by_mode.items():
                update(modes[name], validator, rows, truth, path.name)
        if number % 20 == 0:
            print(f'Processed {number}/92 images in {time.monotonic()-started:.1f}s', flush=True)
    report = {}
    for name, metrics in modes.items():
        metrics.process()
        assert list(map(int, metrics.nt_per_class)) == [89, 8, 36]
        report[name] = dict(overall_AP50=float(metrics.box.map50),
                            per_class={NAMES[int(c)]: dict(AP50=float(metrics.box.ap50[j]),
                            AP50_95=float(metrics.box.ap[j]), precision=float(metrics.box.p[j]),
                            recall=float(metrics.box.r[j])) for j, c in enumerate(metrics.box.ap_class_index)})
        report[name]['all_classes_over_80_AP50'] = all(r['AP50'] >= .8 for r in report[name]['per_class'].values())
    report['runtime_seconds'] = time.monotonic()-started
    report['scope'] = 'Identical 92 validation images and all 133 targets. Fixed image-grid crops; no labels used for crop selection.'
    (OUT / 'tiled_metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
