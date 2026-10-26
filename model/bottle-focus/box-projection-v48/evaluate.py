"""Saved-model evaluations with exact model/cohort/settings receipts."""
from pathlib import Path
import gc
import torch
from ultralytics import YOLO,__version__
import data
import protocol
RUN=data.RUN;ROOT=data.ROOT

def cohorts(audit):
 return [dict(name='retention',yaml=str(RUN/'data.yaml'),photos=audit['retention_photos'],boxes=audit['retention_boxes'],fingerprint=audit['retention_fingerprint'],dataset_sha=data.sha(RUN/'dataset.json'),previously_seen_by_initial_and_v42=True),
         dict(name='openimages',yaml=str(ROOT/'monitor-campaign-v34/data.yaml'),photos=36,boxes=17,fingerprint='1662f8caeec4275e7fdca7c26e89bed6c346227800743fdc20937310e8f0e3d5',dataset_sha=data.sha(ROOT/'monitor-campaign-v34/dataset.json')),
         dict(name='litter',yaml=str(ROOT/'source-coverage-diagnostic-v40/data.yaml'),photos=32,boxes=18,fingerprint='acf8fd5f6133981c8aae103cf343d9fd04728fc069d23436b5849fbc90b84dd3',dataset_sha=data.sha(ROOT/'source-coverage-diagnostic-v40/dataset.json'))]

def evaluate(weights,cohort):
 weights=Path(weights);digest=data.sha(weights);snapshot=data.CACHE/'weights'/f'verified-{digest}.pt';data.parent.clone(weights,snapshot)
 path=RUN/f"score-{cohort['name']}-{digest}.json"
 expected=dict(weights_sha=digest,cohort=cohort,dataset_sha=data.sha(RUN/'dataset.json'),declaration_sha=data.sha(RUN/'declaration.json'),imgsz=640,batch=4,nms_iou=.5,test_time_augmentation=False,device='mps',ultralytics_version=__version__)
 if path.exists():return protocol.receipt(data.load(path),expected)
 model=YOLO(str(snapshot));assert model.names=={0:'bottle'}
 metrics=model.val(data=cohort['yaml'],classes=[0],imgsz=640,batch=4,device='mps',workers=0,augment=False,iou=.5,save_json=True,plots=False,project=str(RUN/'evaluations'),name=f"{cohort['name']}-{digest[:12]}")
 assert list(metrics.box.ap_class_index)==[0] and int(metrics.nt_per_class[0])==cohort['boxes']
 p,r,a,b=metrics.box.class_result(0);report=dict(expected,weights=str(snapshot),AP50=float(a),AP50_95=float(b),precision=float(p),recall=float(r),independent_NHS_test=False)
 assert data.sha(weights)==data.sha(snapshot)==digest;protocol.receipt(report,expected);data.save(path,report);print('CLASSIFIER_EVALUATION',cohort['name'],digest,report['AP50'],report['AP50_95'],flush=True)
 del model;gc.collect();torch.mps.empty_cache();return report
