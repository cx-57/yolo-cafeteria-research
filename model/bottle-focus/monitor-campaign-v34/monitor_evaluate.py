"""Fresh saved-model evaluation on the continuation monitor, never benchmark scoring."""
from pathlib import Path
import gc
import shutil
import torch
from ultralytics import YOLO,__version__

def evaluate(run,weights,role,selected_epoch,common,audit,weights_dir):
 s=common.s;digest=s.sha(weights);snapshot=weights_dir/f'verified-{digest}.pt'
 if not snapshot.exists():
  temporary=snapshot.with_suffix('.pt.partial');shutil.copyfile(weights,temporary);assert s.sha(temporary)==digest;temporary.replace(snapshot)
 assert s.sha(snapshot)==digest
 path=run/f'monitor-{role}.json'
 if path.exists():
  previous=s.load(path)
  assert previous['weights_sha']==digest and previous['monitor_fingerprint']==audit['monitor_fingerprint']
  assert previous['training_dataset_audit_sha']==s.sha(run/'dataset.json') and previous['metric_source']=='continuation_monitor'
  assert previous['imgsz']==640 and previous['batch']==4 and previous['nms_iou']==.5 and previous['test_time_augmentation'] is False
  assert previous['monitor_images']==audit['monitor_images'] and previous['monitor_bottle_boxes']==audit['monitor_bottle_boxes'] and previous['selected_epoch']==selected_epoch
  return previous
 model=YOLO(str(snapshot));metrics=model.val(data=str(run/'data.yaml'),classes=[0],imgsz=640,batch=4,device='mps',workers=0,augment=False,iou=.5,save_json=True,plots=False,project=str(run/'monitor-evaluations'),name=role)
 assert list(metrics.box.ap_class_index)==[0] and int(metrics.nt_per_class[0])==audit['monitor_bottle_boxes']
 p,r,ap50,ap95=metrics.box.class_result(0)
 report=dict(metric_source='continuation_monitor',role=role,selected_epoch=selected_epoch,weights=str(snapshot),weights_sha=digest,AP50=float(ap50),AP50_95=float(ap95),precision=float(p),recall=float(r),monitor_fingerprint=audit['monitor_fingerprint'],monitor_images=audit['monitor_images'],monitor_bottle_boxes=audit['monitor_bottle_boxes'],training_dataset_audit_sha=s.sha(run/'dataset.json'),imgsz=640,batch=4,nms_iou=.5,test_time_augmentation=False,device='mps',ultralytics_version=__version__,fixed_benchmark_used=False,independent_NHS_test=False,declaration_sha=s.sha(run/'declaration.json'))
 assert s.sha(weights)==s.sha(snapshot)==digest;s.save(path,report)
 print('CONTINUATION_MONITOR_RESULT',role,report['AP50'],report['AP50_95'],digest,flush=True)
 del model;gc.collect();torch.mps.empty_cache()
 return report
