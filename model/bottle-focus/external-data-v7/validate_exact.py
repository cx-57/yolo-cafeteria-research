"""Independently reload and verify a selected external-data trial checkpoint."""
from pathlib import Path
import argparse,importlib.util,json,shutil
from ultralytics import YOLO,__version__
parser=argparse.ArgumentParser()
parser.add_argument('--run',type=Path,default=Path(__file__).resolve().parent)
parser.add_argument('--nms-iou',type=float,default=.5)
parser.add_argument('--candidate',choices=['selected','trained'],default='selected')
args=parser.parse_args();RUN=args.run.resolve()
spec=importlib.util.spec_from_file_location('bottle_common',RUN.parent/'plastic-specialist-s-v2/run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);s=m.s
build_spec=importlib.util.spec_from_file_location('external_dataset',RUN/'build.py');dataset=importlib.util.module_from_spec(build_spec);build_spec.loader.exec_module(dataset)
audit=dataset.verify();selection=s.load(RUN/('trained-selection.json' if args.candidate=='trained' else 'selection.json'))
weights=RUN/('best_trained_bottle.pt' if args.candidate=='trained' else 'best_bottle.pt');before=s.sha(weights);assert before==selection['weights_sha']
tag=f'independent-{args.candidate}-tta640-iou{args.nms_iou}'
metrics=YOLO(str(weights)).val(data=str(m.DATA/'data.yaml'),classes=[0],imgsz=640,batch=1,device='mps',workers=0,augment=True,iou=args.nms_iou,save_json=True,plots=False,project=str(RUN/'evaluations'),name=tag)
assert list(metrics.box.ap_class_index)==[0] and int(metrics.nt_per_class[0])==67
p,r,ap,ap95=metrics.box.class_result(0)
report=dict(weights=str(weights),weights_sha=before,AP50=float(ap),AP50_95=float(ap95),precision=float(p),recall=float(r),imgsz=640,batch=1,nms_iou=args.nms_iou,test_time_augmentation=True,validation_images=246,bottle_objects=67,source_dataset_fingerprint=m.MANIFEST['source_fingerprint'],validation_protocol=m.MANIFEST['validation_protocol'],independent_test=False,ultralytics_version=__version__,training_dataset_audit_sha=s.sha(RUN/'dataset.json'),selected_epoch=selection['epoch'],external_training_used=selection['epoch']>0,external_human_verified=False)
assert s.sha(weights)==before
dataset.verify();s.save(RUN/(tag+'.json'),report)
incumbent=RUN.parent/'verified-incumbent'
if report['AP50']>s.load(incumbent/'verification.json')['AP50']:
    source=RUN/f"verified-{args.candidate}-epoch{selection['epoch']}-bottle.pt";shutil.copy2(weights,source)
    shutil.copy2(source,incumbent/'bottle.pt')
    cache=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/weights/bottle-best-verified.pt');shutil.copy2(source,cache)
    assert s.sha(source)==s.sha(incumbent/'bottle.pt')==s.sha(cache)==before
    report.update(weights=str(incumbent/'bottle.pt'),source_checkpoint=str(source),source_verification=str(RUN/(tag+'.json')),runtime_cache_copy=str(cache))
    s.save(incumbent/'verification.json',report)
print('INDEPENDENT_BOTTLE_RESULT',json.dumps(report),flush=True)
