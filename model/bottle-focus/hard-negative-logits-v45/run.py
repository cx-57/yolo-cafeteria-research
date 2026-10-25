"""Three-epoch classifier logit calibration with geometry and retention protection."""
from pathlib import Path
import gc
import shutil
import torch
from ultralytics import YOLO
import data
import geometry
import evaluate
from trainer import Trainer
RUN=data.RUN;ROOT=data.ROOT;load=data.load;save=data.save;sha=data.sha
SOURCE=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-reviewed-neck-monitor-v36/weights/verified-88877bbf13c02a202811b6ca0a1ccdc2bde70b2916d366f4b67938df732d1518.pt')
INITIAL_SHA='88877bbf13c02a202811b6ca0a1ccdc2bde70b2916d366f4b67938df732d1518'
RECIPE=dict(epochs=3,batch=4,nbs=16,imgsz=640,device='mps',workers=0,optimizer='AdamW',lr0=1e-4,warmup_bias_lr=0.,warmup_epochs=1.,seed=43,patience=0,freeze=23,plots=False,save=True,mosaic=0.,close_mosaic=0,scale=.25,translate=.05,cos_lr=True)
WEIGHTS=data.CACHE/'weights'
def check_declaration():
 d=load(RUN/'declaration.json');assert d['recipe']==RECIPE and d['initial_weights_sha']==sha(SOURCE)==INITIAL_SHA
 assert d['dataset_sha']==sha(RUN/'dataset.json') and d['data_yaml_sha']==sha(RUN/'data.yaml')
 assert d['retention_review_sha']==sha(RUN/'retention-review.json') and d['hard_review_sha']==sha(RUN/'hard-review.json')
 assert d['target_scope']=='disposable_plastic_drink_bottles' and d['user_scope_sha']==sha(RUN/'user-scope.json')
 assert d['code_shas']=={n:sha(RUN/n) for n in d['code_shas']} and d['reference_shas']=={n:sha(ROOT/n) for n in d['reference_shas']}
 return d

def recover_terminal(initial):
 progress_path=RUN/'progress.json';stop_path=RUN/'retention-stop.json'
 if not progress_path.exists():
  assert not stop_path.exists(),'Stop marker without checkpoint progress'
  return False
 point=load(progress_path);last=data.CACHE/'train/weights/last.pt'
 assert point['declaration_sha']==sha(RUN/'declaration.json') and sha(last)==point['weights_sha']
 assert 1<=point['epoch']<=RECIPE['epochs']
 stopped=point['epoch']>=2 and (point['AP50']<initial['AP50']-.03 or point['AP50_95']<initial['AP50_95']-.03)
 if stop_path.exists():
  stop=load(stop_path);assert all(stop[k]==point[k] for k in ['epoch','weights_sha','declaration_sha'])
  assert stopped,'Stop marker conflicts with declared retention rule'
 if stopped or point['epoch']>=RECIPE['epochs']:
  save(RUN/'training-complete.json',dict(declaration_sha=sha(RUN/'declaration.json'),progress=point,recovered_terminal_checkpoint=True))
  return True
 return False

def main():
 torch.set_num_threads(2);torch.mps.set_per_process_memory_fraction(.5);check_declaration();authorization=load(RUN/'authorization.json')
 assert authorization['declaration_sha']==sha(RUN/'declaration.json') and authorization['optimization_authorized'] is True and authorization['maximum_epochs_authorized']==20
 audit=data.verify();locks=data.parent.sources.module('v44_locks',ROOT/'monitor-campaign-v34/selection.py');locks.bind_campaign(RUN/'measurement-start.json',sha(RUN/'declaration.json'))
 retention=evaluate.cohorts(audit)[0]
 initial=evaluate.evaluate(SOURCE,retention)
 save(RUN/'retention-diagnostic.json',dict(initial=initial,previously_seen_by_starting_model=True,independent_NHS_test=False,original_benchmark_inference=False))
 if (RUN/'training-complete.json').exists() or recover_terminal(initial):
  import finish;finish.main();return
 assert shutil.disk_usage(RUN).free>=2*1024**3,'Need2GiBfree'
 locks.bind_campaign(RUN/'training-start.json',sha(RUN/'declaration.json'));WEIGHTS.mkdir(parents=True,exist_ok=True)
 initial_model=YOLO(str(SOURCE)).model.float().cpu().eval();geometry.select(initial_model);reference=geometry.fixed_state(initial_model);del initial_model
 def check(trainer):
  names={n for n,p in trainer.model.named_parameters() if p.requires_grad};assert names==geometry.allowed_names(trainer.model) and len(names)==6
  assert not any(m.training for m in trainer.model.modules() if isinstance(m,torch.nn.BatchNorm2d))
 def completed(trainer):
  geometry.verify_fixed(trainer.model,reference);geometry.verify_fixed(trainer.ema.ema,reference)
  _,_,a,b=trainer.validator.metrics.box.class_result(0);point=dict(epoch=trainer.epoch+1,AP50=float(a),AP50_95=float(b),metric_source='source_matched_retention_monitor',retention_fingerprint=audit['retention_fingerprint'],declaration_sha=sha(RUN/'declaration.json'))
  assert 1<=point['epoch']<=RECIPE['epochs'] and 0<=point['AP50_95']<=point['AP50']<=1
  saved=YOLO(str(trainer.last)).model.float().cpu().eval();geometry.select(saved);geometry.verify_fixed(saved,reference);del saved
  save(RUN/'geometry-proof.json',dict(epoch=point['epoch'],fixed_tensor_fingerprint=geometry.fingerprint(reference),training_model_fixed=True,EMA_fixed=True,saved_checkpoint_fixed=True,classifier_logits_only=True,mutable_parameter_tensors=6,classifier_features_fixed=True,all_BatchNorm_buffers_fixed=True))
  for role,key in [('ap50','AP50'),('localization','AP50_95')]:
   path=RUN/f'native-{role}.json';old=load(path) if path.exists() else None
   if old is None or point[key]>old[key]:
    shutil.copyfile(trainer.last,WEIGHTS/f'best-{role}.pt');save(path,dict(point,weights_sha=sha(WEIGHTS/f'best-{role}.pt')))
  save(RUN/'progress.json',dict(point,weights_sha=sha(trainer.last)))
  if point['epoch']>=2 and (point['AP50']<initial['AP50']-.03 or point['AP50_95']<initial['AP50_95']-.03):
   trainer.stop=True;save(RUN/'retention-stop.json',dict(point,weights_sha=sha(trainer.last),reason='Retention regression exceeds3percentage points'))
  print('CLASSIFIER_RETENTION_EPOCH',point,flush=True)
 resume=data.CACHE/'train/weights/last.pt';model=YOLO(str(resume if resume.exists() else SOURCE))
 model.add_callback('on_train_batch_start',check);model.add_callback('on_model_save',completed)
 model.train(trainer=Trainer,resume=resume.exists(),data=str(RUN/'data.yaml'),classes=[0],project=str(data.CACHE),name='train',exist_ok=True,**RECIPE)
 del model;gc.collect();torch.mps.empty_cache();save(RUN/'training-complete.json',dict(declaration_sha=sha(RUN/'declaration.json'),progress=load(RUN/'progress.json')))
 import finish;finish.main()
if __name__=='__main__':main()
