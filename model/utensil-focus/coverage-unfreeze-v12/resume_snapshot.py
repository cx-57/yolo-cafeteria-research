from pathlib import Path
import json,time,shutil,gc
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
import utensil_packet as s
root=Path.cwd();data=root/'data/utensil-only/coverage-v12';run=root/'model/utensil-focus/coverage-unfreeze-v12';manifest=s.load(data/'manifest.json');s.validate(manifest['records'],data)
checkpoint=torch.load(run/'last_resume.pt',map_location='cpu',weights_only=False)
assert checkpoint['optimizer'] is not None and checkpoint['epoch']>=0
print('Resuming after epoch',checkpoint['epoch']+1,flush=True);del checkpoint
class Trainer(DetectionTrainer):
 def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
  return super().get_dataloader(dataset_path,4 if mode=='val' else batch_size,rank,mode)
 def final_eval(self):pass
started=time.monotonic()
def completed(t):
 score=float(t.validator.metrics.box.ap50[0]);print('UTENSIL_EPOCH',t.epoch+1,'AP50',score,flush=True);shutil.copy2(t.last,run/'last_resume.pt')
 if score>s.load(run/'selection.json')['AP50']:
  shutil.copy2(t.last,run/'best_utensil.pt');s.save(run/'selection.json',dict(epoch=t.epoch+1,AP50=score,weights_sha=s.sha(run/'best_utensil.pt')))
 if time.monotonic()-started>=360 or score>=.90:t.stop=True
model=YOLO(str(run/'last_resume.pt'));model.add_callback('on_model_save',completed);model.train(trainer=Trainer,resume=True)
del model;gc.collect();torch.mps.empty_cache()
for tag,weights in [('selected',run/'best_utensil.pt'),('latest',run/'train/weights/last.pt')]:
 model=YOLO(str(weights));metrics=model.val(data=str(data/'data.yaml'),classes=[0],imgsz=640,batch=4,device='mps',workers=0,plots=False,project=str(run/'evaluations'),name=tag)
 p,r,a,b=metrics.box.class_result(0);report=dict(weights=str(weights),weights_sha=s.sha(weights),dataset_fingerprint=manifest['fingerprint'],target='utensil',classes=[0],imgsz=640,test_time_augmentation=False,independent_test=False,precision=float(p),recall=float(r),AP50=float(a),AP50_95=float(b));s.save(run/(tag+'.json'),report);print(json.dumps(report),flush=True)
 del model;gc.collect();torch.mps.empty_cache()
s.validate(manifest['records'],data)
