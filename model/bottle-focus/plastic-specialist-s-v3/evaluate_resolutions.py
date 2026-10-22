"""Evaluate the saved incumbent at larger inference resolutions, unchanged val."""
from pathlib import Path
import gc
import json
import torch
from ultralytics import YOLO
import utensil_packet as s

RUN = Path(__file__).resolve().parent
DATA = Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-capture-validation')
manifest = s.load(DATA / 'manifest.json')
s.NAMES = ['bottle']
s.validate_configuration(DATA, manifest['records'])
s.validate(manifest['records'], DATA)
torch.set_num_threads(2)
weights = RUN / 'best_bottle.pt'
for imgsz, augment, iou in [(960, False, .7), (960, True, .5), (1280, True, .5)]:
    tag = f'resolution-{imgsz}-tta{int(augment)}-iou{iou}'
    if (RUN / (tag + '.json')).exists():
        continue
    model = YOLO(str(weights))
    metrics = model.val(data=str(DATA / 'data.yaml'), split='val', classes=[0],
                        imgsz=imgsz, batch=1, device='mps', workers=0,
                        augment=augment, iou=iou, save_json=True, plots=False,
                        project=str(RUN / 'evaluations'), name=tag)
    assert list(metrics.box.ap_class_index) == [0] and int(metrics.nt_per_class[0]) == 67
    p, r, ap, ap95 = metrics.box.class_result(0)
    report = dict(weights=str(weights), weights_sha=s.sha(weights), AP50=float(ap),
                  AP50_95=float(ap95), precision=float(p), recall=float(r),
                  imgsz=imgsz, batch=1, test_time_augmentation=augment, iou=iou,
                  validation_images=246, bottle_objects=67,
                  source_dataset_fingerprint=manifest['source_fingerprint'],
                  validation_protocol=manifest['validation_protocol'], independent_test=False)
    s.save(RUN / (tag + '.json'), report)
    print('BOTTLE_RESOLUTION', json.dumps(report), flush=True)
    del model
    gc.collect()
    torch.mps.empty_cache()
    if ap >= .85:
        break
s.validate(manifest['records'], DATA)
