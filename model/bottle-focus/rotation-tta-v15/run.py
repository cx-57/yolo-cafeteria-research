"""Evaluate one fixed four-orientation TTA configuration on pinned bottle weights."""
from pathlib import Path
import importlib.util,json,shutil,types
from ultralytics import YOLO,__version__
from rotation_tta import full_resolution_rotations
RUN=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('bottle_dataset',RUN.parent/'external-data-v8/build.py');dataset=importlib.util.module_from_spec(spec);spec.loader.exec_module(dataset)
audit=dataset.verify();m=dataset.m;s=m.s
weights=RUN.parent/'head-average-v9/averaged-bottle.pt'
proof=s.load(RUN/'geometry-verification.json');before=s.sha(weights)
assert before==proof['weights_sha']=='42079f5ac4167c9c1f42627df4f6b175943b3ebff102d5ba342af48958f73571'
assert s.sha(RUN/'rotation_tta.py')==proof['adapter_sha'] and s.sha(RUN/'test_rotation_tta.py')==proof['test_sha']
assert proof['tests_passed']==2
averaging=s.load(RUN.parent/'head-average-v9/averaging.json')
assert s.sha(RUN.parent/'external-data-v8/dataset.json')==averaging['training_dataset_audit_sha']
trace=dict(calls=0,shapes=set())
def traced_augment(self,x):
    trace['calls']+=1;trace['shapes'].add(tuple(x.shape))
    return full_resolution_rotations(self,x)
model=YOLO(str(weights));model.model._predict_augment=types.MethodType(traced_augment,model.model)
tag='rotation4-640-iou0.5'
metrics=model.val(data=str(m.DATA/'data.yaml'),classes=[0],imgsz=640,batch=1,device='mps',workers=0,augment=True,iou=.5,save_json=True,plots=False,project=str(RUN/'evaluations'),name=tag)
assert list(metrics.box.ap_class_index)==[0] and int(metrics.nt_per_class[0])==67
assert trace['calls']==246,trace
p,r,ap,ap95=metrics.box.class_result(0)
report=dict(weights=str(weights),weights_sha=before,AP50=float(ap),AP50_95=float(ap95),precision=float(p),recall=float(r),imgsz=640,batch=1,nms_iou=.5,test_time_augmentation=True,tta_mode='Four full-resolution rotations: 0,90,180,270 degrees; confidence scores unchanged; ordinary NMS.',rotation_degrees=[0,90,180,270],rotation_adapter=str(RUN/'rotation_tta.py'),rotation_adapter_sha=proof['adapter_sha'],geometry_verification=proof,augmentation_calls=trace['calls'],input_shapes=sorted(trace['shapes']),validation_images=246,bottle_objects=67,source_dataset_fingerprint=m.MANIFEST['source_fingerprint'],validation_protocol=m.MANIFEST['validation_protocol'],averaging=averaging,independent_test=False,ultralytics_version=__version__)
assert s.sha(weights)==before;dataset.verify();s.save(RUN/(tag+'.json'),report)
incumbent=RUN.parent/'verified-incumbent'
if report['AP50']>s.load(incumbent/'verification.json')['AP50']:
    shutil.copy2(weights,incumbent/'bottle.pt');cache=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/weights/bottle-best-verified.pt');shutil.copy2(weights,cache)
    assert s.sha(incumbent/'bottle.pt')==s.sha(cache)==before
    report.update(weights=str(incumbent/'bottle.pt'),source_checkpoint=str(weights),source_verification=str(RUN/(tag+'.json')),runtime_cache_copy=str(cache));s.save(incumbent/'verification.json',report)
print('ROTATION_BOTTLE_RESULT',json.dumps(report),flush=True)
