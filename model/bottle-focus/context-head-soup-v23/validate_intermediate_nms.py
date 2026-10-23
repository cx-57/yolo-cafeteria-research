"""Freshly verify the incumbent with intermediate NMS IoU 0.6 on MPS."""
from pathlib import Path
import importlib.util,json,shutil
from ultralytics import YOLO,__version__
import torch
torch.set_num_threads(2)
RUN=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('bottle_common',RUN.parent/'plastic-specialist-s-v2/run.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);s=m.s
weights=RUN/'alpha0.5-bottle.pt';before=s.sha(weights)
assert before==s.load(RUN/'alpha0.5-averaging.json')['weights_sha']
metrics=YOLO(str(weights)).val(data=str(m.DATA/'data.yaml'),classes=[0],imgsz=640,batch=4,device='mps',workers=0,augment=False,iou=.6,save_json=True,plots=False,project=str(RUN/'evaluations'),name='ordinary-iou0.6-mps')
assert list(metrics.box.ap_class_index)==[0] and int(metrics.nt_per_class[0])==67
p,r,ap,ap95=metrics.box.class_result(0)
report=dict(weights=str(weights),weights_sha=before,AP50=float(ap),AP50_95=float(ap95),precision=float(p),recall=float(r),imgsz=640,batch=4,nms_iou=.6,device='mps',test_time_augmentation=False,validation_images=246,bottle_objects=67,source_dataset_fingerprint=m.MANIFEST['source_fingerprint'],validation_protocol=m.MANIFEST['validation_protocol'],averaging=s.load(RUN/'alpha0.5-averaging.json'),independent_test=False,ultralytics_version=__version__)
assert before==s.sha(weights)
s.validate_configuration(m.DATA,m.MANIFEST['records']);s.validate(m.MANIFEST['records'],m.DATA)
s.save(RUN/'ordinary-iou0.6-mps.json',report)
incumbent=RUN.parent/'verified-incumbent'
if report['AP50']>s.load(incumbent/'verification.json')['AP50']:
    shutil.copy2(weights,incumbent/'bottle.pt');cache=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/weights/bottle-best-verified.pt');shutil.copy2(weights,cache)
    assert s.sha(incumbent/'bottle.pt')==s.sha(cache)==before
    report.update(weights=str(incumbent/'bottle.pt'),source_checkpoint=str(weights),source_verification=str(RUN/'ordinary-iou0.6-mps.json'),runtime_cache_copy=str(cache));s.save(incumbent/'verification.json',report)
print('ORDINARY_BOTTLE_RESULT',json.dumps(report),flush=True)
