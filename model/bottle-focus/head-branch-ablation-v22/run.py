"""Evaluate two predeclared independent detection-head branch replacements."""
from pathlib import Path
import importlib.util,json,shutil,gc
import torch
from ultralytics import YOLO,__version__
from branch_mix import mix_branches
torch.set_num_threads(2)
RUN=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('bottle_dataset',RUN.parent/'external-data-v8/build.py');dataset=importlib.util.module_from_spec(spec);spec.loader.exec_module(dataset)
m=dataset.m;s=m.s;dataset.verify()
initial=RUN.parent/'head-interpolation-v16/alpha0.75-bottle.pt';quality=RUN.parent/'quality-head-blend-v19/alpha0.25-bottle.pt'
assert s.sha(initial)=='e222b49cb8671bc33d085e82afb77087645924fdb5e9018c63ef9d3a3e0277ec'
assert s.sha(quality)=='88877bbf13c02a202811b6ca0a1ccdc2bde70b2916d366f4b67938df732d1518'
proof=s.load(RUN/'verification.json');assert s.sha(RUN/'branch_mix.py')==proof['implementation_sha'] and s.sha(RUN/'test_branch_mix.py')==proof['test_sha']
for branch in ('regression','classification'):
    tag=f'{branch}-ordinary640-iou0.5-cpu';weights=RUN/f'{branch}-bottle.pt'
    if not weights.exists():
        a=YOLO(str(initial)).model.float().cpu().eval();b=YOLO(str(quality)).model.float().cpu().eval()
        hybrid=mix_branches(a,b,branch);changed=[k for k,v in a.state_dict().items() if not torch.equal(v,hybrid.state_dict()[k])]
        assert changed
        torch.save(dict(model=hybrid.half(),ema=None,optimizer=None,epoch=-1,version=__version__,train_args=dict(task='detect')),weights)
        construction=dict(branch=branch,initial=str(initial),initial_sha=s.sha(initial),quality=str(quality),quality_sha=s.sha(quality),weights_sha=s.sha(weights),changed_state_keys=changed,frozen_backbone_identical=True,BatchNorm_running_statistics_identical=True,quality_averaging=s.load(RUN.parent/'quality-head-blend-v19/alpha0.25-averaging.json'),initial_averaging=s.load(RUN.parent/'head-interpolation-v16/alpha0.75-averaging.json'),implementation_verification=proof,independent_test=False)
        s.save(RUN/f'{branch}-construction.json',construction)
        del a,b,hybrid;gc.collect()
    before=s.sha(weights);assert before==s.load(RUN/f'{branch}-construction.json')['weights_sha']
    model=YOLO(str(weights));metrics=model.val(data=str(m.DATA/'data.yaml'),classes=[0],imgsz=640,batch=4,device='cpu',workers=0,augment=False,iou=.5,save_json=True,plots=False,project=str(RUN/'evaluations'),name=tag)
    assert list(metrics.box.ap_class_index)==[0] and int(metrics.nt_per_class[0])==67
    p,r,ap,ap95=metrics.box.class_result(0)
    report=dict(weights=str(weights),weights_sha=before,AP50=float(ap),AP50_95=float(ap95),precision=float(p),recall=float(r),imgsz=640,batch=4,nms_iou=.5,device='cpu',torch_threads=2,test_time_augmentation=False,validation_images=246,bottle_objects=67,source_dataset_fingerprint=m.MANIFEST['source_fingerprint'],validation_protocol=m.MANIFEST['validation_protocol'],branch_construction=s.load(RUN/f'{branch}-construction.json'),independent_test=False,ultralytics_version=__version__)
    assert s.sha(weights)==before;dataset.verify();s.save(RUN/(tag+'.json'),report)
    incumbent=RUN.parent/'verified-incumbent'
    if report['AP50']>s.load(incumbent/'verification.json')['AP50']:
        shutil.copy2(weights,incumbent/'bottle.pt');cache=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/weights/bottle-best-verified.pt');shutil.copy2(weights,cache)
        assert s.sha(incumbent/'bottle.pt')==s.sha(cache)==before
        report.update(weights=str(incumbent/'bottle.pt'),source_checkpoint=str(weights),source_verification=str(RUN/(tag+'.json')),runtime_cache_copy=str(cache));s.save(incumbent/'verification.json',report)
    print('BRANCH_BOTTLE_RESULT',json.dumps(report),flush=True)
    del model,metrics;gc.collect()
