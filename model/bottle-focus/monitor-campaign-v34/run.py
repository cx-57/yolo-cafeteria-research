"""Bounded head-only continuation; choose on monitor, benchmark one final candidate."""
from pathlib import Path
import gc
import importlib.util
import shutil
import torch
from ultralytics import YOLO

RUN=Path(__file__).resolve().parent

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
m=module('v34_common',RUN.parent/'plastic-specialist-s-v2/run.py');s=m.s
dataset=module('v34_dataset',RUN/'build.py');selection=module('v34_selection',RUN/'selection.py');finisher=module('v34_finish',RUN/'finish.py')
WEIGHTS=dataset.CACHE/'weights';SOURCE=RUN.parent/'quality-head-blend-v19/alpha0.25-bottle.pt';SHA='88877bbf13c02a202811b6ca0a1ccdc2bde70b2916d366f4b67938df732d1518'

def main():
 audit=dataset.verify();declaration=s.load(RUN/'declaration.json')
 assert declaration['max_epochs']==selection.MAX_EPOCHS==9 and declaration['monitor_fingerprint']==audit['monitor_fingerprint']
 assert declaration['training_dataset_audit_sha']==s.sha(RUN/'dataset.json') and declaration['selection_code_sha']==s.sha(RUN/'selection.py') and declaration['partition_code_sha']==s.sha(RUN/'partition.py')
 assert declaration['code_shas']=={name:s.sha(RUN/name) for name in declaration['code_shas']}
 selection.bind_campaign(RUN/'campaign-start.json',s.sha(RUN/'declaration.json'))
 if (RUN/'campaign-result.json').exists():finisher.finish(m,dataset,WEIGHTS);return
 WEIGHTS.mkdir(parents=True,exist_ok=True);assert s.sha(SOURCE)==SHA
 if not (WEIGHTS/'initial.pt').exists():shutil.copyfile(SOURCE,WEIGHTS/'initial.pt')
 assert s.sha(WEIGHTS/'initial.pt')==SHA
 if (RUN/'monitor-progress.json').exists() and selection.budget_finished(s.load(RUN/'monitor-progress.json')['epoch']):finisher.finish(m,dataset,WEIGHTS);return
 if shutil.disk_usage(RUN).free<1024**3:raise RuntimeError('Less than1GiB free before monitored specialist trial')
 if not (RUN/'monitor-initial.json').exists():finisher.monitor_eval.evaluate(RUN,WEIGHTS/'initial.pt','initial',0,m,audit,WEIGHTS)
 def completed(trainer):
  _,_,a,b=trainer.validator.metrics.box.class_result(0);point=dict(epoch=trainer.epoch+1,AP50=float(a),AP50_95=float(b),metric_source='continuation_monitor',monitor_fingerprint=audit['monitor_fingerprint'])
  finisher.shared.metrics(point);assert 1<=point['epoch']<=9
  shutil.copyfile(trainer.last,WEIGHTS/'last_resume.pt');s.save(RUN/'monitor-progress.json',point)
  for role,key in [('ap50','AP50'),('localization','AP50_95')]:
   path=RUN/f'native-{role}-selection.json';old=s.load(path) if path.exists() else None
   if old is None or point[key]>old[key]:
    destination=WEIGHTS/f'best-monitor-{role}.pt';shutil.copyfile(trainer.last,destination);s.save(path,dict(point,weights_sha=s.sha(destination)))
  print('MONITOR_EPOCH',point,flush=True)
  if selection.budget_finished(point['epoch']):trainer.stop=True
 def check(trainer):
  if getattr(trainer,'v34_checked_epoch',None)==trainer.epoch:return
  trainable=[name for name,p in trainer.model.named_parameters() if p.requires_grad];assert trainable and all(name.startswith('model.23.') for name in trainable)
  norms=[layer for layer in trainer.model.modules() if isinstance(layer,torch.nn.BatchNorm2d)];assert norms and not any(layer.training for layer in norms)
  s.save(RUN/'trainability.json',dict(epoch=trainer.epoch+1,head_only=True,all_BatchNorm_statistics_fixed=True,normalization_modules=len(norms),native_validation_source='continuation_monitor'))
  trainer.v34_checked_epoch=trainer.epoch
 resume=(WEIGHTS/'last_resume.pt').exists();model=YOLO(str(WEIGHTS/'last_resume.pt' if resume else WEIGHTS/'initial.pt'));assert model.model.yaml['scale']=='s' and model.model.model[-1].nc==1
 model.add_callback('on_train_batch_start',check);model.add_callback('on_model_save',completed)
 model.train(trainer=m.Trainer,resume=resume,data=str(RUN/'data.yaml'),classes=[0],epochs=9,batch=4,nbs=16,imgsz=640,device='mps',workers=0,optimizer='AdamW',lr0=.00001,warmup_bias_lr=0.,warmup_epochs=1.,seed=43,patience=0,freeze=23,plots=False,project=str(dataset.CACHE),name='train',save=True,mosaic=0.,close_mosaic=0,scale=.25,translate=.05)
 del model;gc.collect();torch.mps.empty_cache();finisher.finish(m,dataset,WEIGHTS)

if __name__=='__main__':main()
