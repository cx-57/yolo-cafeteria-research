"""Three pinned saved-model diagnostic evaluations; no training or benchmark calls."""
from pathlib import Path
import gc
import hashlib
import importlib.util
import json
import torch
from ultralytics import YOLO,__version__

RUN=Path(__file__).resolve().parent;ROOT=RUN.parent
RUNTIME=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data')
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
dataset=module('v40_dataset',RUN/'build.py');protocol=module('v40_protocol',RUN/'protocol.py');locks=module('v40_locks',ROOT/'monitor-campaign-v34/selection.py')
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()

def models():
 rows=[('balanced','bottle-reviewed-neck-monitor-v36','88877bbf13c02a202811b6ca0a1ccdc2bde70b2916d366f4b67938df732d1518','monitor-campaign-v34/monitor-initial.json'),
       ('v34','bottle-monitor-v34','641d162e111ad497a36717ababf5da912cda3de38c79ab8b1443ebe5555ba942','monitor-campaign-v34/monitor-ap50.json'),
       ('v36','bottle-reviewed-neck-monitor-v36','4afd31080d871eea5993aade1fb871697cdc009e5b2f2f77c75e3b11d5c41e93','reviewed-neck-monitor-v36/monitor-ap50.json')]
 return [dict(role=role,weights=str(RUNTIME/folder/'weights'/f'verified-{digest}.pt'),weights_sha=digest,old_monitor_reference=str(ROOT/reference),old_monitor_reference_sha=sha(ROOT/reference)) for role,folder,digest,reference in rows]

def evaluate(record,audit):
 path=RUN/f"{record['role']}-litter.json";data_sha=sha(RUN/'dataset.json');weights=Path(record['weights']);assert sha(weights)==record['weights_sha']
 if path.exists():
  report=json.loads(path.read_text());assert report['role']==record['role'] and report['old_monitor_reference_sha']==record['old_monitor_reference_sha']
  assert report['declaration_sha']==sha(RUN/'declaration.json') and report['ultralytics_version']==__version__ and report['device']=='mps'
  return protocol.validate_report(report,record['weights_sha'],data_sha,audit['cohort_fingerprint'],audit['bottle_objects'])
 model=YOLO(str(weights));assert model.names=={0:'bottle'}
 metrics=model.val(data=str(RUN/'data.yaml'),classes=[0],imgsz=640,batch=4,device='mps',workers=0,augment=False,iou=.5,save_json=True,plots=False,project=str(RUN/'evaluations'),name=record['role'])
 assert list(metrics.box.ap_class_index)==[0] and int(metrics.nt_per_class[0])==audit['bottle_objects']
 p,r,ap,ap95=metrics.box.class_result(0)
 report=dict(record,dataset_sha=data_sha,cohort_fingerprint=audit['cohort_fingerprint'],metric_source='fresh_litter_diagnostic',photos=32,bottle_objects=audit['bottle_objects'],
             AP50=float(ap),AP50_95=float(ap95),precision=float(p),recall=float(r),imgsz=640,batch=4,nms_iou=.5,test_time_augmentation=False,
             device='mps',ultralytics_version=__version__,original_benchmark_used=False,independent_NHS_test=False,full_training_started=False,declaration_sha=sha(RUN/'declaration.json'))
 assert sha(weights)==record['weights_sha'];protocol.validate_report(report,record['weights_sha'],data_sha,audit['cohort_fingerprint'],audit['bottle_objects'])
 path.write_text(json.dumps(report,indent=2)+'\n');print('LITTER_DIAGNOSTIC_RESULT',record['role'],report['AP50'],report['AP50_95'],flush=True)
 del model;gc.collect();torch.mps.empty_cache();return report

def main():
 torch.set_num_threads(2);audit=dataset.verify();declaration=json.loads((RUN/'declaration.json').read_text());book=models()
 assert declaration['models']==book and declaration['dataset_sha']==sha(RUN/'dataset.json') and declaration['data_yaml_sha']==sha(RUN/'data.yaml')
 assert declaration['cohort_fingerprint']==audit['cohort_fingerprint'] and declaration['model_evaluation_limit']==3 and declaration['original_benchmark_inference_allowed'] is False
 assert declaration['code_shas']=={name:sha(RUN/name) for name in declaration['code_shas']}
 locks.bind_campaign(RUN/'measurement-start.json',sha(RUN/'declaration.json'))
 reports=[]
 for record in book:reports.append(evaluate(record,audit))
 dataset.verify();old=[]
 for record in book:
  reference=json.loads(Path(record['old_monitor_reference']).read_text());assert sha(record['old_monitor_reference'])==record['old_monitor_reference_sha'] and reference['weights_sha']==record['weights_sha']
  assert reference['metric_source']=='continuation_monitor' and reference['monitor_fingerprint']=='1662f8caeec4275e7fdca7c26e89bed6c346227800743fdc20937310e8f0e3d5'
  old.append(dict(role=record['role'],AP50=reference['AP50'],AP50_95=reference['AP50_95'],metric_source='original_Open_Images_monitor',reference=record['old_monitor_reference'],reference_sha=record['old_monitor_reference_sha']))
 result=dict(status='source_diagnostic_complete_not_original_benchmark_success',litter_reports=reports,old_monitor_references=old,
             dataset_sha=sha(RUN/'dataset.json'),cohort_fingerprint=audit['cohort_fingerprint'],original_benchmark_used=False,
             may_satisfy_user_move_on_criterion=False,model_promotion=False,full_training_started=False,independent_NHS_test=False)
 (RUN/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print('SOURCE_DIAGNOSTIC_COMPLETE',flush=True)

if __name__=='__main__':main()
