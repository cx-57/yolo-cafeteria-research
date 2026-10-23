"""Freshly score immutable candidate bytes at one fixed ordinary-inference configuration."""
from pathlib import Path
import gc
import shutil
import torch
from ultralytics import YOLO,__version__

def evaluate(run,weights,role,epoch,common,dataset,policy):
    s=common.s;dataset.verify();digest=s.sha(weights)
    snapshot=run/f'verified-{digest}.pt'
    if not snapshot.exists():
        temporary=snapshot.with_suffix('.pt.partial');shutil.copyfile(weights,temporary)
        assert s.sha(temporary)==digest;temporary.replace(snapshot)
    assert s.sha(snapshot)==digest
    model=YOLO(str(snapshot))
    measured=model.val(data=str(common.DATA/'data.yaml'),classes=[0],imgsz=640,batch=4,device='mps',workers=0,augment=False,iou=.5,save_json=True,plots=False,project=str(run/'evaluations'),name=f'phase{epoch}-{role}')
    assert list(measured.box.ap_class_index)==[0] and int(measured.nt_per_class[0])==67
    p,r,ap50,ap95=measured.box.class_result(0)
    report=dict(weights=str(snapshot),weights_sha=digest,source_checkpoint=str(weights),role=role,completed_epoch=epoch,AP50=float(ap50),AP50_95=float(ap95),precision=float(p),recall=float(r),imgsz=640,batch=4,nms_iou=.5,device='mps',test_time_augmentation=False,validation_images=246,bottle_objects=67,source_dataset_fingerprint=common.MANIFEST['source_fingerprint'],validation_protocol=common.MANIFEST['validation_protocol'],training_dataset_audit_sha=s.sha(run/'dataset.json'),ultralytics_version=__version__,independent_test=False,external_human_verified=False)
    policy.metrics(report);assert s.sha(weights)==s.sha(snapshot)==digest;dataset.verify()
    path=run/f'phase{epoch}-{role}.json';s.save(path,report)
    history=policy.verified_records(run);history.append(report)
    s.save(run/'metrics-history.json',history)
    s.save(run/'frontier.json',dict(target_AP50=.90,minimum_AP50_95=.64,aspirational_AP50_95=.70,members=policy.frontier(history)))
    incumbent=run.parent/'verified-incumbent';old=s.load(incumbent/'verification.json')
    if report['AP50']>=old['AP50'] and report['AP50_95']>=old['AP50_95'] and (report['AP50']>old['AP50'] or report['AP50_95']>old['AP50_95']):
        cache=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/weights/bottle-best-verified.pt')
        shutil.copyfile(snapshot,incumbent/'bottle.pt');shutil.copyfile(snapshot,cache)
        assert s.sha(incumbent/'bottle.pt')==s.sha(cache)==digest
        s.save(incumbent/'verification.json',dict(report,weights=str(incumbent/'bottle.pt'),source_verification=str(path),runtime_cache_copy=str(cache)))
    print('JOINT_METRIC_RESULT',role,report['AP50'],report['AP50_95'],digest,flush=True)
    del model;gc.collect();torch.mps.empty_cache()
    return report
