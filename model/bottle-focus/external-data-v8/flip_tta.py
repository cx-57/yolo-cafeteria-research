"""Test same-resolution horizontal-flip inference without shrinking small bottles."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,shutil,types
import torch
from ultralytics import YOLO,__version__

def full_resolution_flip(self,x):
    original=self._predict_once(x)[0]
    reflected=self._predict_once(x.flip(3))[0]
    reflected=self._descale_pred(reflected,3,1.,x.shape[-2:])
    return torch.cat((original,reflected),-1),None

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--probe-only',action='store_true')
    args=parser.parse_args()
    RUN=Path(__file__).resolve().parent;weights=RUN/'phase6-best_bottle.pt'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    before=sha(weights)
    assert before==json.loads((RUN/'phase6-selected.json').read_text())['weights_sha']
    torch.set_num_threads(2)
    model=YOLO(str(weights));native=model.model.float().cpu().eval()
    assert native.model[-1].nc==1 and not getattr(native,'end2end',False)
    state={k:v.clone() for k,v in native.state_dict().items()}
    torch.manual_seed(42);x=torch.rand(1,3,96,128)
    with torch.inference_mode():
        plain=native._predict_once(x)[0].clone()
        flip=native._predict_once(x.flip(3))[0].clone()
        combined,_=full_resolution_flip(native,x)
    count=plain.shape[-1]
    torch.testing.assert_close(combined[...,:count],plain,rtol=0,atol=0)
    expected=flip.clone();expected[:,0]=x.shape[-1]-flip[:,0]
    torch.testing.assert_close(combined[...,count:],expected,rtol=0,atol=0)
    for k,v in native.state_dict().items():torch.testing.assert_close(v,state[k],rtol=0,atol=0)
    proof=dict(weights_sha=before,selected_epoch=int(model.ckpt['epoch'])+1,identity_prediction_preserved_exactly=True,flip_coordinates_scores_preserved_exactly=True,model_state_unchanged=True,probe_shape=list(x.shape),tta='original + horizontal flip, both scale 1.0; no multi-scale tail clipping')
    (RUN/'flip-tta-probe.json').write_text(json.dumps(proof,indent=2)+'\n')
    print('FLIP_TTA_PROBE',json.dumps(proof),flush=True)
    del model,native,state,plain,flip,combined,x
    if args.probe_only:raise SystemExit(0)
    spec=importlib.util.spec_from_file_location('dataset',RUN/'build.py');dataset=importlib.util.module_from_spec(spec);spec.loader.exec_module(dataset)
    audit=dataset.verify();m=dataset.m;s=m.s
    model=YOLO(str(weights));model.model._predict_augment=types.MethodType(full_resolution_flip,model.model)
    tag='full-resolution-flip640-iou0.5'
    metrics=model.val(data=str(m.DATA/'data.yaml'),classes=[0],imgsz=640,batch=1,device='mps',workers=0,augment=True,iou=.5,save_json=True,plots=False,project=str(RUN/'evaluations'),name=tag)
    assert list(metrics.box.ap_class_index)==[0] and int(metrics.nt_per_class[0])==67
    p,r,ap,ap95=metrics.box.class_result(0)
    report=dict(weights=str(weights),weights_sha=before,AP50=float(ap),AP50_95=float(ap95),precision=float(p),recall=float(r),imgsz=640,batch=1,nms_iou=.5,test_time_augmentation=True,tta_mode=proof['tta'],validation_images=246,bottle_objects=67,source_dataset_fingerprint=m.MANIFEST['source_fingerprint'],validation_protocol=m.MANIFEST['validation_protocol'],training_dataset_audit_sha=s.sha(RUN/'dataset.json'),selected_epoch=proof['selected_epoch'],external_training_used=True,external_human_verified=False,independent_test=False,ultralytics_version=__version__)
    assert sha(weights)==before;dataset.verify();s.save(RUN/(tag+'.json'),report)
    incumbent=RUN.parent/'verified-incumbent'
    if report['AP50']>s.load(incumbent/'verification.json')['AP50']:
        shutil.copy2(weights,incumbent/'bottle.pt')
        cache=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/weights/bottle-best-verified.pt');shutil.copy2(weights,cache)
        assert sha(incumbent/'bottle.pt')==sha(cache)==before
        report.update(weights=str(incumbent/'bottle.pt'),source_checkpoint=str(weights),source_verification=str(RUN/(tag+'.json')),runtime_cache_copy=str(cache))
        s.save(incumbent/'verification.json',report)
    print('FULL_RESOLUTION_FLIP_RESULT',json.dumps(report),flush=True)
