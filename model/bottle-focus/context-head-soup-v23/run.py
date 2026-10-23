"""Evaluate a fixed original-context snapshot and its half average with the incumbent."""
from pathlib import Path
import importlib.util,json,shutil,gc
import torch
from ultralytics import YOLO,__version__
torch.set_num_threads(2)
RUN=Path(__file__).resolve().parent
DATA_RUN=RUN.parent/'external-data-v8'
spec=importlib.util.spec_from_file_location('bottle_dataset',DATA_RUN/'build.py');dataset=importlib.util.module_from_spec(spec);spec.loader.exec_module(dataset)
m=dataset.m;s=m.s;dataset.verify()
quality_spec=importlib.util.spec_from_file_location('quality_dataset',RUN.parent/'context-balance-v20/build.py');quality=importlib.util.module_from_spec(quality_spec);quality_spec.loader.exec_module(quality);quality.verify()
snapshot=s.load(RUN/'source-snapshot.json')
initial=RUN.parent/'quality-head-blend-v19/alpha0.25-bottle.pt';trained=Path(snapshot['snapshot'])
assert s.sha(RUN.parent/'context-balance-v20/dataset.json')==snapshot['training_dataset_audit_sha']
assert s.sha(initial)=='88877bbf13c02a202811b6ca0a1ccdc2bde70b2916d366f4b67938df732d1518'
assert s.sha(trained)==snapshot['selection']['weights_sha']
for alpha in (1.,.5):
    tag=f'alpha{alpha}-ordinary640-iou0.5-cpu';weights=RUN/f'alpha{alpha}-bottle.pt'
    if not weights.exists():
        a=YOLO(str(initial)).model.float().cpu().eval();b=YOLO(str(trained)).model.float().cpu().eval()
        state_a,state_b=a.state_dict(),b.state_dict();assert state_a.keys()==state_b.keys();changed=[]
        for key,value in state_a.items():
            other=state_b[key]
            if not torch.equal(value,other):
                assert key.startswith('model.23.') and not any(t in key for t in ('running_mean','running_var','num_batches_tracked')),key
                changed.append(key)
            if value.is_floating_point():
                value.copy_(other) if alpha==1. else value.lerp_(other,alpha)
            else:assert torch.equal(value,other)
        assert changed;a.load_state_dict(state_a,strict=True)
        if alpha==1.:
            for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key],rtol=0,atol=0)
        torch.save(dict(model=a.half(),ema=None,optimizer=None,epoch=-1,version=__version__,train_args=dict(task='detect')),weights)
        s.save(RUN/f'alpha{alpha}-averaging.json',dict(alpha=alpha,initial=str(initial),initial_sha=s.sha(initial),trained=str(trained),trained_sha=s.sha(trained),weights_sha=s.sha(weights),changed_state_keys=changed,frozen_backbone_identical=True,BatchNorm_running_statistics_identical=True,training_dataset_audit_shas=[s.sha(DATA_RUN/'dataset.json'),s.sha(RUN.parent/'hard-negative-polish-v11/dataset.json'),s.sha(RUN.parent/'context-balance-v20/dataset.json')],initial_verification=s.load(RUN.parent/'quality-head-blend-v19/alpha0.25-ordinary640-iou0.5.json'),trained_snapshot=snapshot,external_training_used=True,external_human_verified=False,independent_test=False))
        del a,b,state_a,state_b;gc.collect()
    before=s.sha(weights);assert before==s.load(RUN/f'alpha{alpha}-averaging.json')['weights_sha']
    model=YOLO(str(weights));metrics=model.val(data=str(m.DATA/'data.yaml'),classes=[0],imgsz=640,batch=4,device='cpu',workers=0,augment=False,iou=.5,save_json=True,plots=False,project=str(RUN/'evaluations'),name=tag)
    assert list(metrics.box.ap_class_index)==[0] and int(metrics.nt_per_class[0])==67
    p,r,ap,ap95=metrics.box.class_result(0)
    report=dict(weights=str(weights),weights_sha=before,AP50=float(ap),AP50_95=float(ap95),precision=float(p),recall=float(r),imgsz=640,batch=4,nms_iou=.5,device='cpu',test_time_augmentation=False,validation_images=246,bottle_objects=67,source_dataset_fingerprint=m.MANIFEST['source_fingerprint'],validation_protocol=m.MANIFEST['validation_protocol'],averaging=s.load(RUN/f'alpha{alpha}-averaging.json'),independent_test=False,ultralytics_version=__version__)
    assert s.sha(weights)==before;dataset.verify();s.save(RUN/(tag+'.json'),report)
    incumbent=RUN.parent/'verified-incumbent'
    if report['AP50']>s.load(incumbent/'verification.json')['AP50']:
        shutil.copy2(weights,incumbent/'bottle.pt');cache=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/weights/bottle-best-verified.pt');shutil.copy2(weights,cache)
        assert s.sha(incumbent/'bottle.pt')==s.sha(cache)==before
        report.update(weights=str(incumbent/'bottle.pt'),source_checkpoint=str(weights),source_verification=str(RUN/(tag+'.json')),runtime_cache_copy=str(cache));s.save(incumbent/'verification.json',report)
    print('INTERPOLATION_BOTTLE_RESULT',json.dumps(report),flush=True)
    del model,metrics;gc.collect()
