"""Compare two fixed neighboring head averages with ordinary saved-model inference."""
from pathlib import Path
import importlib.util,json,shutil,gc
import torch
from ultralytics import YOLO,__version__
RUN=Path(__file__).resolve().parent
DATA_RUN=RUN.parent/'external-data-v8'
spec=importlib.util.spec_from_file_location('bottle_dataset',DATA_RUN/'build.py');dataset=importlib.util.module_from_spec(spec);spec.loader.exec_module(dataset)
m=dataset.m;s=m.s;dataset.verify()
initial=DATA_RUN/'initial.pt';trained=DATA_RUN/'phase6-best_bottle.pt'
assert s.sha(initial)=='edab60c58fb8a1e941557b4bd84955a5208b75b476e0a587854d3e679e7d1c63'
assert s.sha(trained)=='082bf8188cac9449d9d20cdad52cefd57c6d1125992dcc860d0428ab5a47b0e2'
for alpha in (.25,.75):
    tag=f'alpha{alpha}-ordinary640-iou0.5';weights=RUN/f'alpha{alpha}-bottle.pt'
    if not weights.exists():
        a=YOLO(str(initial)).model.float().cpu().eval();b=YOLO(str(trained)).model.float().cpu().eval()
        state_a,state_b=a.state_dict(),b.state_dict();assert state_a.keys()==state_b.keys();changed=[]
        for key,value in state_a.items():
            other=state_b[key]
            if not torch.equal(value,other):
                assert key.startswith('model.23.') and not any(t in key for t in ('running_mean','running_var','num_batches_tracked')),key
                changed.append(key)
            if value.is_floating_point():value.lerp_(other,alpha)
            else:assert torch.equal(value,other)
        assert changed;a.load_state_dict(state_a,strict=True)
        torch.save(dict(model=a.half(),ema=None,optimizer=None,epoch=-1,version=__version__,train_args=dict(task='detect')),weights)
        s.save(RUN/f'alpha{alpha}-averaging.json',dict(alpha=alpha,initial=str(initial),initial_sha=s.sha(initial),trained=str(trained),trained_sha=s.sha(trained),weights_sha=s.sha(weights),changed_state_keys=changed,frozen_backbone_identical=True,BatchNorm_running_statistics_identical=True,training_dataset_audit_sha=s.sha(DATA_RUN/'dataset.json'),external_training_used=True,external_human_verified=False,independent_test=False))
        del a,b,state_a,state_b;gc.collect()
    before=s.sha(weights);assert before==s.load(RUN/f'alpha{alpha}-averaging.json')['weights_sha']
    model=YOLO(str(weights));metrics=model.val(data=str(m.DATA/'data.yaml'),classes=[0],imgsz=640,batch=4,device='mps',workers=0,augment=False,iou=.5,save_json=True,plots=False,project=str(RUN/'evaluations'),name=tag)
    assert list(metrics.box.ap_class_index)==[0] and int(metrics.nt_per_class[0])==67
    p,r,ap,ap95=metrics.box.class_result(0)
    report=dict(weights=str(weights),weights_sha=before,AP50=float(ap),AP50_95=float(ap95),precision=float(p),recall=float(r),imgsz=640,batch=4,nms_iou=.5,test_time_augmentation=False,validation_images=246,bottle_objects=67,source_dataset_fingerprint=m.MANIFEST['source_fingerprint'],validation_protocol=m.MANIFEST['validation_protocol'],averaging=s.load(RUN/f'alpha{alpha}-averaging.json'),independent_test=False,ultralytics_version=__version__)
    assert s.sha(weights)==before;dataset.verify();s.save(RUN/(tag+'.json'),report)
    incumbent=RUN.parent/'verified-incumbent'
    if report['AP50']>s.load(incumbent/'verification.json')['AP50']:
        shutil.copy2(weights,incumbent/'bottle.pt');cache=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/weights/bottle-best-verified.pt');shutil.copy2(weights,cache)
        assert s.sha(incumbent/'bottle.pt')==s.sha(cache)==before
        report.update(weights=str(incumbent/'bottle.pt'),source_checkpoint=str(weights),source_verification=str(RUN/(tag+'.json')),runtime_cache_copy=str(cache));s.save(incumbent/'verification.json',report)
    print('INTERPOLATION_BOTTLE_RESULT',json.dumps(report),flush=True)
    del model,metrics;gc.collect();torch.mps.empty_cache()
