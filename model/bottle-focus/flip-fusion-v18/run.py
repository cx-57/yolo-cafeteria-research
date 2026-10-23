"""Freshly evaluate one fixed two-view WBF configuration on saved bottle weights."""
from pathlib import Path
import importlib.util,json,shutil,types,sys
from ultralytics import YOLO,__version__
from fusion import FlipFusionValidator
RUN=Path(__file__).resolve().parent
sys.path.insert(0,str(RUN.parent/'external-data-v8'))
from flip_tta import full_resolution_flip
spec=importlib.util.spec_from_file_location('bottle_dataset',RUN.parent/'external-data-v8/build.py');dataset=importlib.util.module_from_spec(spec);spec.loader.exec_module(dataset)
audit=dataset.verify();m=dataset.m;s=m.s
weights=RUN.parent/'head-interpolation-v16/alpha0.75-bottle.pt'
proof=s.load(RUN/'verification.json');before=s.sha(weights)
assert before=='e222b49cb8671bc33d085e82afb77087645924fdb5e9018c63ef9d3a3e0277ec'
assert s.sha(RUN/'fusion.py')==proof['fusion_sha'] and s.sha(RUN/'test_fusion.py')==proof['fusion_test_sha']
assert s.sha(RUN/'vendor_wbf.py')==proof['vendor_sha']==s.load(RUN/'source.json')['sha256']
assert s.sha(RUN.parent/'external-data-v8/flip_tta.py')==proof['flip_sha']
averaging=s.load(RUN.parent/'head-interpolation-v16/alpha0.75-averaging.json')
assert s.sha(RUN.parent/'external-data-v8/dataset.json')==averaging['training_dataset_audit_sha']
trace=dict(flip_calls=0,shapes=set())
def traced_augment(self,x):
    trace['flip_calls']+=1;trace['shapes'].add(tuple(x.shape))
    return full_resolution_flip(self,x)
def completed(validator):
    trace['fused_images']=validator.fused_images
model=YOLO(str(weights));model.model._predict_augment=types.MethodType(traced_augment,model.model)
model.add_callback('on_val_end',completed)
tag='flip2-wbf640'
metrics=model.val(validator=FlipFusionValidator,data=str(m.DATA/'data.yaml'),classes=[0],imgsz=640,batch=1,device='mps',workers=0,augment=True,iou=.5,save_json=True,plots=False,project=str(RUN/'evaluations'),name=tag)
assert list(metrics.box.ap_class_index)==[0] and int(metrics.nt_per_class[0])==67
assert trace['flip_calls']==trace['fused_images']==246,trace
p,r,ap,ap95=metrics.box.class_result(0)
report=dict(weights=str(weights),weights_sha=before,AP50=float(ap),AP50_95=float(ap95),precision=float(p),recall=float(r),imgsz=640,batch=1,nms_iou=.5,test_time_augmentation=True,tta_mode='Original plus horizontal flip at full resolution, per-view NMS and Weighted Boxes Fusion.',image_views=['original','horizontal_flip'],wbf=dict(iou_thr=.55,skip_box_thr=.001,conf_type='avg',weights=[1,1],allows_overflow=False),runtime_script=str(RUN/'run.py'),runtime_script_sha=s.sha(RUN/'run.py'),implementation_verification=proof,augmentation_calls=trace['flip_calls'],fused_images=trace['fused_images'],input_shapes=sorted(trace['shapes']),validation_images=246,bottle_objects=67,source_dataset_fingerprint=m.MANIFEST['source_fingerprint'],validation_protocol=m.MANIFEST['validation_protocol'],averaging=averaging,independent_test=False,ultralytics_version=__version__)
assert s.sha(weights)==before;dataset.verify();s.save(RUN/(tag+'.json'),report)
incumbent=RUN.parent/'verified-incumbent'
if report['AP50']>s.load(incumbent/'verification.json')['AP50']:
    shutil.copy2(weights,incumbent/'bottle.pt');cache=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/weights/bottle-best-verified.pt');shutil.copy2(weights,cache)
    assert s.sha(incumbent/'bottle.pt')==s.sha(cache)==before
    report.update(weights=str(incumbent/'bottle.pt'),source_checkpoint=str(weights),source_verification=str(RUN/(tag+'.json')),runtime_cache_copy=str(cache));s.save(incumbent/'verification.json',report)
print('FUSION_BOTTLE_RESULT',json.dumps(report),flush=True)
