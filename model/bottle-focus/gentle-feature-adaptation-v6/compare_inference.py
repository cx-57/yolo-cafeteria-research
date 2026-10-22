"""Compare three explicit inference settings on the immutable phase-8 detector."""
from pathlib import Path
import gc
import importlib.util
import json
import shutil
import torch
from ultralytics import YOLO, __version__

RUN = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('bottle_common', RUN.parent / 'plastic-specialist-s-v2/run.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = m.s
weights = RUN / 'phase8-best_bottle.pt'
proof = s.load(RUN / 'phase8-selected-tta640.json')
assert s.sha(weights) == proof['weights_sha']

if __name__ == '__main__':
    settings = [(832, False, .7), (832, True, .7), (640, True, .5)]
    for size, augment, nms_iou in settings:
        tag = f'inference-{size}-tta{int(augment)}-iou{nms_iou}'
        if (RUN / f'{tag}.json').exists():
            continue
        model = YOLO(str(weights))
        metrics = model.val(data=str(m.DATA / 'data.yaml'), classes=[0], imgsz=size,
                            batch=1 if augment else 4, device='mps', workers=0,
                            augment=augment, iou=nms_iou, save_json=True, plots=False,
                            project=str(RUN / 'evaluations'), name=tag)
        assert list(metrics.box.ap_class_index) == [0] and int(metrics.nt_per_class[0]) == 67
        p, r, ap, ap95 = metrics.box.class_result(0)
        report = dict(proof, weights=str(weights), AP50=float(ap), AP50_95=float(ap95),
                      precision=float(p), recall=float(r), imgsz=size,
                      batch=1 if augment else 4, nms_iou=nms_iou,
                      test_time_augmentation=augment, ultralytics_version=__version__,
                      source_verification=str(RUN / f'{tag}.json'))
        assert s.sha(weights) == report['weights_sha']
        s.validate_configuration(m.DATA, m.MANIFEST['records'])
        s.validate(m.MANIFEST['records'], m.DATA)
        s.save(RUN / f'{tag}.json', report)
        incumbent = RUN.parent / 'verified-incumbent'
        if report['AP50'] > s.load(incumbent / 'verification.json')['AP50']:
            shutil.copy2(weights, incumbent / 'bottle.pt')
            report.update(weights=str(incumbent / 'bottle.pt'), source_checkpoint=str(weights),
                          runtime_cache_copy='/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/weights/bottle-best-verified.pt')
            shutil.copy2(weights, report['runtime_cache_copy'])
            s.save(incumbent / 'verification.json', report)
        print('BOTTLE_INFERENCE', tag, json.dumps(report), flush=True)
        del model, metrics
        gc.collect()
        torch.mps.empty_cache()
