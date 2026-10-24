"""Train reviewed full images and neck adaptation from the stronger localization specialist and verify both metrics."""
from pathlib import Path
import argparse
import gc
import importlib.util
import os
import shutil
import torch
from ultralytics import YOLO

RUN=Path(__file__).resolve().parent
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
m=module('v31_common',RUN.parent/'plastic-specialist-s-v2/run.py');s=m.s
dataset=module('v31_data',RUN/'build.py');policy=module('v31_policy',RUN/'policy.py');evaluator=module('v31_evaluation',RUN/'evaluate.py')
SOURCE=RUN.parent/'quality-head-blend-v19/alpha0.25-bottle.pt'
PROOF=RUN.parent/'quality-head-blend-v19/alpha0.25-ordinary640-iou0.5.json'
guard=module('v31_trainability',RUN/'trainability.py')
SHA='88877bbf13c02a202811b6ca0a1ccdc2bde70b2916d366f4b67938df732d1518'
ROLES=dict(ap50=('best_trained_bottle.pt','trained-selection.json'),localization=('best_localization_bottle.pt','localization-selection.json'),joint=('best_joint_bottle.pt','joint-selection.json'))

def initialize():
    proof=s.load(PROOF);assert s.sha(SOURCE)==proof['weights_sha']==SHA
    assert proof['source_dataset_fingerprint']==m.MANIFEST['source_fingerprint'] and proof['validation_protocol']==m.MANIFEST['validation_protocol']
    initial=RUN/'initial.pt'
    if not initial.exists():shutil.copyfile(SOURCE,initial)
    assert s.sha(initial)==SHA
    s.save(RUN/'initialization.json',dict(source=str(SOURCE),source_verification=str(PROOF),weights_sha=SHA,baseline_AP50=proof['AP50'],baseline_AP50_95=proof['AP50_95']))

def evaluate_phase():
    epoch=s.load(RUN/'progress.json')['epoch']
    roles=[(role,RUN/paths[0]) for role,paths in ROLES.items() if (RUN/paths[1]).exists()]
    roles.append(('latest',RUN/'last_resume.pt'));seen=set()
    for role,path in roles:
        digest=s.sha(path)
        if digest in seen:continue
        seen.add(digest);evaluator.evaluate(RUN,path,role,epoch,m,dataset,policy)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--evaluate-only',action='store_true');args=parser.parse_args()
    audit=dataset.verify();initialize()
    if not (RUN/'metrics-history.json').exists():evaluator.evaluate(RUN,RUN/'initial.pt','initial',0,m,dataset,policy)
    if args.evaluate_only:
        evaluate_phase();raise SystemExit(0)
    resume=(RUN/'last_resume.pt').exists();start=s.load(RUN/'progress.json')['epoch'] if resume else 0
    phase_epochs=int(os.environ.get('BOTTLE_PHASE_EPOCHS','3'))
    s.save(RUN/'experiment.json',dict(target_AP50=.90,minimum_AP50_95=.64,aspirational_AP50_95=.70,initial_weights_sha=SHA,data=str(RUN/'data.yaml'),epochs=40,phase_epochs=phase_epochs,batch=4,nbs=16,imgsz=640,freeze=11,optimizer='AdamW',lr0=.00001,seed=43,mosaic=.5,scale=.5,close_mosaic=0,normalization='All BatchNorm running statistics fixed',original_training_photos=audit['training_unique_images'],original_training_bottle_boxes=audit['training_unique_bottle_boxes'],training_exposures=audit['training_exposures'],cropped_exposures=0,independent_test=False))
    def completed(trainer):
        _,_,ap50,ap95=trainer.validator.metrics.box.class_result(0)
        point=dict(epoch=trainer.epoch+1,AP50=float(ap50),AP50_95=float(ap95))
        print('BOTTLE_EPOCH',point,flush=True)
        shutil.copyfile(trainer.last,RUN/'last_resume.pt');s.save(RUN/'progress.json',point)
        previous={role:s.load(RUN/files[1]) for role,files in ROLES.items() if (RUN/files[1]).exists()}
        for role in policy.save_roles(point,previous):
            weights,selection=ROLES[role];shutil.copyfile(trainer.last,RUN/weights)
            s.save(RUN/selection,dict(point,weights_sha=s.sha(RUN/weights)))
        if policy.target_met(point) or trainer.epoch+1>=start+phase_epochs or (RUN/'stop-after-epoch').exists():trainer.stop=True
    def trainability(trainer):
        if getattr(trainer,'v31_checked_epoch',None)==trainer.epoch:return
        s.save(RUN/'trainability.json',dict(epoch=trainer.epoch+1,**guard.audit(trainer.model)))
        trainer.v31_checked_epoch=trainer.epoch
    model=YOLO(str(RUN/'last_resume.pt' if resume else RUN/'initial.pt'))
    assert model.model.yaml['scale']=='s' and model.model.model[-1].nc==1
    model.add_callback('on_train_batch_start',trainability);model.add_callback('on_model_save',completed)
    model.train(trainer=m.Trainer,resume=resume,data=str(RUN/'data.yaml'),classes=[0],epochs=40,batch=4,imgsz=640,device='mps',workers=0,optimizer='AdamW',lr0=.00001,warmup_bias_lr=0.,warmup_epochs=1.,nbs=16,seed=43,patience=0,freeze=11,plots=False,project=str(RUN),name='train',save=True,mosaic=.5,close_mosaic=0,scale=.5)
    del model;gc.collect();torch.mps.empty_cache();evaluate_phase()
