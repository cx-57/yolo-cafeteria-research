from pathlib import Path
import copy,json,time,shutil,gc
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
import utensil_packet as s
root=Path.cwd();run=root/'model/utensil-focus/coco-union-finetune-v14';data=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/utensil-v12');manifest=s.load(data/'manifest.json')
s.validate(manifest['records'],data)
assert s.sha(run/'initial.pt')==s.load(run/'initialization.json')['weights_sha']
class Trainer(DetectionTrainer):
 def get_model(self,cfg=None,weights=None,verbose=True):
  assert weights.model[-1].nc==1 and isinstance(weights.model[-1].cv3[0][-1],torch.nn.Sequential)
  return copy.deepcopy(weights).float()
 def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
  return super().get_dataloader(dataset_path,4 if mode=='val' else batch_size,rank,mode)
 def _model_train(self):
  super()._model_train()
  for m in self.model.modules():
   if isinstance(m,torch.nn.BatchNorm2d):m.eval()
 def final_eval(self):pass

def evaluate(weights,tag,augment=False):
 model=YOLO(str(weights));metrics=model.val(data=str(data/'utensil.yaml'),classes=[0],imgsz=640,batch=1 if augment else 4,device='mps',workers=0,augment=augment,save_json=True,plots=False,project=str(run/'evaluations'),name=tag)
 p,r,a,b=metrics.box.class_result(0);report=dict(weights=str(weights),weights_sha=s.sha(weights),dataset_fingerprint=manifest['fingerprint'],target='utensil',classes=[0],imgsz=640,test_time_augmentation=augment,independent_test=False,precision=float(p),recall=float(r),AP50=float(a),AP50_95=float(b));s.save(run/(tag+'.json'),report);print('V14_RESULT',tag,json.dumps(report),flush=True)
 del model;gc.collect();torch.mps.empty_cache();return report
baseline=evaluate(run/'initial.pt','baseline');shutil.copy2(run/'initial.pt',run/'best_utensil.pt');s.save(run/'selection.json',dict(epoch=0,AP50=baseline['AP50'],weights_sha=s.sha(run/'best_utensil.pt')))
s.save(run/'experiment.json',dict(target='utensil',dataset_fingerprint=manifest['fingerprint'],epochs=40,phase_minutes=6,lr0=.00005,nbs=32,freeze=23,BatchNorm_running_stats='Fixed at pretrained values; affine parameters may learn in trainable detection head',mosaic=0,selection='Highest ordinary utensil AP50 with initialization as incumbent',validation_images=764,validation_utensil_boxes=296,independent_test=False))
started=time.monotonic()
def completed(t):
 score=float(t.validator.metrics.box.ap50[0]);print('UTENSIL_EPOCH',t.epoch+1,'AP50',score,flush=True);shutil.copy2(t.last,run/'last_resume.pt')
 if score>s.load(run/'selection.json')['AP50']:
  shutil.copy2(t.last,run/'best_utensil.pt');s.save(run/'selection.json',dict(epoch=t.epoch+1,AP50=score,weights_sha=s.sha(run/'best_utensil.pt')))
 if time.monotonic()-started>=360 or score>=.90:t.stop=True
model=YOLO(str(run/'initial.pt'));model.add_callback('on_model_save',completed)
model.train(trainer=Trainer,data=str(data/'utensil.yaml'),classes=[0],epochs=40,batch=4,imgsz=640,device='mps',workers=0,optimizer='AdamW',lr0=.00005,warmup_bias_lr=0.,warmup_epochs=1.,nbs=32,seed=42,patience=0,freeze=23,plots=False,project=str(run),name='train',save=True,mosaic=0.,mixup=0.,cutmix=0.,copy_paste=0.,close_mosaic=0)
del model;gc.collect();torch.mps.empty_cache()
evaluate(run/'best_utensil.pt','selected');evaluate(run/'best_utensil.pt','selected-tta640',True);s.validate(manifest['records'],data)
