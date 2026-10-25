"""Full bottle training; launch only after user lifts their earlier pause."""
from pathlib import Path
import gc
import json
import shutil
import torch
from ultralytics import YOLO
import build
import protocol
RUN=Path(__file__).resolve().parent;ROOT=RUN.parent
sha=build.sha;save=build.save;load=lambda p:json.loads(Path(p).read_text())
WEIGHTS=build.CACHE/'weights'
SOURCE=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-reviewed-neck-monitor-v36/weights/verified-88877bbf13c02a202811b6ca0a1ccdc2bde70b2916d366f4b67938df732d1518.pt')
INITIAL_SHA='88877bbf13c02a202811b6ca0a1ccdc2bde70b2916d366f4b67938df732d1518'
RECIPE=dict(epochs=20,batch=4,nbs=16,imgsz=640,device='mps',workers=0,optimizer='AdamW',lr0=1e-5,warmup_bias_lr=0.,warmup_epochs=1.,seed=43,patience=10,freeze=0,plots=False,save=True,mosaic=.3,close_mosaic=10,scale=.5,translate=.1,cos_lr=True)
def trainability(model):
 trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
 counts={role:sum(p.numel() for n,p in trainable if lo<=int(n.split('.')[1])<hi) for role,lo,hi in [('backbone',0,11),('neck',11,23),('head',23,24)]}
 assert all(counts.values())
 norms=[m for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d)]
 assert norms and not any(m.training for m in norms)
 return dict(trainable_parameters=counts,BatchNorm_modules=len(norms),BatchNorm_statistics_fixed=True)
def declaration_check():
 d=load(RUN/'declaration.json');assert d['recipe']==RECIPE and d['initial_weights_sha']==sha(SOURCE)==INITIAL_SHA
 assert d['dataset_sha']==sha(RUN/'dataset.json') and d['data_yaml_sha']==sha(RUN/'data.yaml')
 assert d['code_shas']=={name:sha(RUN/name) for name in d['code_shas']}
 assert d['reference_shas']=={name:sha(ROOT/name) for name in d['reference_shas']}
 return d

def main():
 declaration_check()
 receipt=load(RUN/'authorization.json') if (RUN/'authorization.json').exists() else None
 protocol.validate_authorization(receipt,sha(RUN/'declaration.json'))
 audit=build.verify()
 if (RUN/'training-complete.json').exists():
  finish=build.sources.module('v42_resume_finish',RUN/'finish.py');finish.main();return
 if shutil.disk_usage(RUN).free<2*1024**3:raise RuntimeError('At least2GiB free required before full training')
 common=build.sources.module('v42_common',ROOT/'plastic-specialist-s-v2/run.py')
 locks=build.sources.module('v42_locks',ROOT/'monitor-campaign-v34/selection.py')
 locks.bind_campaign(RUN/'training-start.json',sha(RUN/'declaration.json'));WEIGHTS.mkdir(parents=True,exist_ok=True)
 def check(trainer):
  if getattr(trainer,'v42_epoch',None)!=trainer.epoch:
   save(RUN/'trainability.json',dict(trainability(trainer.model),epoch=trainer.epoch+1));trainer.v42_epoch=trainer.epoch
 def completed(trainer):
  _,_,a,b=trainer.validator.metrics.box.class_result(0)
  point=dict(epoch=trainer.epoch+1,AP50=float(a),AP50_95=float(b),metric_source='combined_development_monitor')
  assert 1<=point['epoch']<=RECIPE['epochs'] and 0<=point['AP50_95']<=point['AP50']<=1
  for role,key in [('ap50','AP50'),('localization','AP50_95')]:
   path=RUN/f'native-{role}.json';old=load(path) if path.exists() else None
   if old is None or point[key]>old[key]:
    shutil.copyfile(trainer.last,WEIGHTS/f'best-{role}.pt');save(path,dict(point,weights_sha=sha(WEIGHTS/f'best-{role}.pt')))
  save(RUN/'progress.json',dict(point,weights_sha=sha(trainer.last)))
  print('FULL_TRAINING_EPOCH',point,flush=True)
 resume=build.CACHE/'train/weights/last.pt'
 model=YOLO(str(resume if resume.exists() else SOURCE));assert model.names=={0:'bottle'} and model.model.yaml['scale']=='s'
 model.add_callback('on_train_batch_start',check);model.add_callback('on_model_save',completed)
 model.train(trainer=common.Trainer,resume=resume.exists(),data=str(RUN/'data.yaml'),classes=[0],project=str(build.CACHE),name='train',exist_ok=True,**RECIPE)
 del model;gc.collect();torch.mps.empty_cache()
 save(RUN/'training-complete.json',dict(declaration_sha=sha(RUN/'declaration.json'),progress=load(RUN/'progress.json')))
 finish=build.sources.module('v42_finish',RUN/'finish.py');finish.main()
if __name__=='__main__':main()
