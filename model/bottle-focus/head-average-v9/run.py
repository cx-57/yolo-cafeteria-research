"""Test one fixed half/half average of corresponding bottle detection heads."""
from pathlib import Path
import gc,importlib.util,json,shutil
import torch
from ultralytics import YOLO,__version__
RUN=Path(__file__).resolve().parent
DATA_RUN=RUN.parent/'external-data-v8'
spec=importlib.util.spec_from_file_location('dataset',DATA_RUN/'build.py');dataset=importlib.util.module_from_spec(spec);spec.loader.exec_module(dataset)
m=dataset.m;s=m.s;m.RUN=RUN
audit=dataset.verify()
initial=DATA_RUN/'initial.pt';trained=DATA_RUN/'phase6-best_bottle.pt'
assert s.sha(initial)==s.load(DATA_RUN/'initialization.json')['weights_sha']
assert s.sha(trained)==s.load(DATA_RUN/'phase6-selected.json')['weights_sha']
weights=RUN/'averaged-bottle.pt'
if not weights.exists():
    a=YOLO(str(initial)).model.float().cpu().eval();b=YOLO(str(trained)).model.float().cpu().eval()
    state_a,state_b=a.state_dict(),b.state_dict();assert state_a.keys()==state_b.keys()
    changed=[]
    for key,value in state_a.items():
        other=state_b[key]
        if not torch.equal(value,other):
            assert key.startswith('model.23.'),key
            assert not any(t in key for t in ('running_mean','running_var','num_batches_tracked')),key
            changed.append(key)
        if value.is_floating_point():value.lerp_(other,.5)
        else:assert torch.equal(value,other)
    assert changed
    a.load_state_dict(state_a,strict=True)
    torch.save(dict(model=a.half(),ema=None,optimizer=None,epoch=-1,version=__version__,train_args=dict(task='detect')),weights)
    s.save(RUN/'averaging.json',dict(alpha=.5,initial=str(initial),initial_sha=s.sha(initial),trained=str(trained),trained_sha=s.sha(trained),weights_sha=s.sha(weights),changed_state_keys=changed,frozen_backbone_identical=True,BatchNorm_running_statistics_identical=True,training_dataset_audit_sha=s.sha(DATA_RUN/'dataset.json'),external_training_used=True,external_human_verified=False,independent_test=False))
    del a,b,state_a,state_b
    gc.collect()
for tag,augment in [('averaged',False),('averaged-tta640',True)]:
    model=YOLO(str(weights));metrics=model.val(data=str(m.DATA/'data.yaml'),classes=[0],imgsz=640,batch=1 if augment else 4,device='mps',workers=0,augment=augment,iou=.5 if augment else .7,save_json=True,plots=False,project=str(RUN/'evaluations'),name=tag)
    assert list(metrics.box.ap_class_index)==[0] and int(metrics.nt_per_class[0])==67
    p,r,ap,ap95=metrics.box.class_result(0)
    report=dict(weights=str(weights),weights_sha=s.sha(weights),AP50=float(ap),AP50_95=float(ap95),precision=float(p),recall=float(r),imgsz=640,batch=1 if augment else 4,nms_iou=.5 if augment else .7,test_time_augmentation=augment,validation_images=246,bottle_objects=67,source_dataset_fingerprint=m.MANIFEST['source_fingerprint'],validation_protocol=m.MANIFEST['validation_protocol'],training_dataset_audit_sha=s.sha(DATA_RUN/'dataset.json'),averaging=s.load(RUN/'averaging.json'),independent_test=False)
    assert report['weights_sha']==s.load(RUN/'averaging.json')['weights_sha']
    dataset.verify();s.save(RUN/(tag+'.json'),report)
    incumbent=RUN.parent/'verified-incumbent'
    if report['AP50']>s.load(incumbent/'verification.json')['AP50']:
        shutil.copy2(weights,incumbent/'bottle.pt');cache=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/weights/bottle-best-verified.pt');shutil.copy2(weights,cache)
        assert s.sha(incumbent/'bottle.pt')==s.sha(cache)==report['weights_sha']
        report.update(weights=str(incumbent/'bottle.pt'),source_checkpoint=str(weights),source_verification=str(RUN/(tag+'.json')),runtime_cache_copy=str(cache));s.save(incumbent/'verification.json',report)
    print('AVERAGED_BOTTLE_RESULT',tag,json.dumps(report),flush=True)
    del model,metrics
    gc.collect();torch.mps.empty_cache()
    if report['AP50']>=.85:break
