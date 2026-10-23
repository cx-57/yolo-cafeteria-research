"""Run verified three-epoch phases until saved AP50 reaches 85% or the 40-epoch schedule ends."""
from pathlib import Path
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time

RUN=Path(__file__).resolve().parent
REPORT='independent-trained-ordinary640-iou0.5.json'
load=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def verified_phase(run):
    audit=load(run/'dataset.json');progress=load(run/'progress.json')
    selection=load(run/'trained-selection.json');report=load(run/REPORT)
    assert report['weights_sha']==selection['weights_sha']==sha(run/'best_trained_bottle.pt')
    assert report['selected_epoch']==selection['epoch']<=progress['epoch']
    assert report['source_dataset_fingerprint']==audit['fixed_source_fingerprint']
    assert report['validation_protocol']==audit['validation_protocol']
    assert report['training_dataset_audit_sha']==sha(run/'dataset.json')
    assert report['validation_images']==246 and report['bottle_objects']==67
    assert report['imgsz']==640 and report['batch']==4 and report['nms_iou']==.5
    assert report['test_time_augmentation'] is False
    assert math.isfinite(report['AP50']) and 0<=report['AP50']<=1
    return progress['epoch'],report

def main():
    _,prior=verified_phase(RUN.parent/'reviewed-expansion-v28')
    if prior['AP50']>=.85:
        print('TARGET_ALREADY_VERIFIED_BY_V28',prior['AP50'],flush=True);return
    state_path=RUN/'continuation.json'
    state=load(state_path) if state_path.exists() else dict(target_AP50=.85,phase_epochs=3,max_epochs=40,phases=[])
    assert state['target_AP50']==.85 and state['phase_epochs']==3 and state['max_epochs']==40
    state['status']='running'
    state_path.write_text(json.dumps(state,indent=2)+'\n')
    pending=None
    if (RUN/'progress.json').exists():
        try:
            existing=verified_phase(RUN)
        except (AssertionError,KeyError,ValueError,OSError) as error:
            state['status']='verification_required'
            state_path.write_text(json.dumps(state,indent=2)+'\n')
            raise RuntimeError('Interrupted phase needs fresh verification: run validate_exact.py --run this directory --candidate trained --ordinary before restarting') from error
        if existing[1]['AP50']>=.85 or existing[0]>=40:pending=existing
    while True:
        before=load(RUN/'progress.json')['epoch'] if (RUN/'progress.json').exists() else 0
        if pending is not None:
            epoch,report=pending;pending=None
        elif before>=40:
            epoch,report=verified_phase(RUN)
        else:
            log=RUN/f'phase-from-{before}-{time.time_ns()}.log'
            print('START_BOTTLE_PHASE',before,'LOG',str(log),flush=True)
            state['current_phase_log']=str(log)
            state_path.write_text(json.dumps(state,indent=2)+'\n')
            with log.open('w') as output:
                try:
                    subprocess.run([sys.executable,str(RUN/'run.py')],env=dict(os.environ,BOTTLE_PHASE_EPOCHS='3'),stdout=output,stderr=subprocess.STDOUT,check=True)
                except subprocess.CalledProcessError:
                    state['status']='failed'
                    state_path.write_text(json.dumps(state,indent=2)+'\n')
                    raise
            epoch,report=verified_phase(RUN)
            assert before<epoch<=40, 'Training did not advance within its declared schedule'
        previous=next((r for r in state['phases'] if r['weights_sha']==report['weights_sha']),None)
        snapshot=Path(previous['snapshot']) if previous else RUN/f'phase{epoch}-trained-bottle.pt'
        if not snapshot.exists():shutil.copy2(RUN/'best_trained_bottle.pt',snapshot)
        assert sha(snapshot)==report['weights_sha']
        archived=dict(report,weights=str(snapshot),snapshot_epoch=epoch)
        (RUN/f'phase{epoch}-trained-verification.json').write_text(json.dumps(archived,indent=2)+'\n')
        state['phases'].append(dict(completed_epoch=epoch,AP50=report['AP50'],weights_sha=report['weights_sha'],snapshot=str(snapshot)))
        state['status']='target_verified' if report['AP50']>=.85 else 'schedule_complete_below_target' if epoch>=40 else 'running'
        state_path.write_text(json.dumps(state,indent=2)+'\n')
        print('VERIFIED_BOTTLE_PHASE',json.dumps(state['phases'][-1]),state['status'],flush=True)
        if state['status']!='running':return

if __name__=='__main__':main()
