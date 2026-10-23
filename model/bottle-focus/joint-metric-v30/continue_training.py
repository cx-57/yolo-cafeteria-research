"""Bounded joint-metric screening; plateaus, storage limits and schedule ends are not success."""
from pathlib import Path
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

RUN=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v30_policy',RUN/'policy.py');policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
load=lambda p:json.loads(p.read_text())

def main():
    path=RUN/'continuation.json'
    state=load(path) if path.exists() else dict(target_AP50=.90,minimum_AP50_95=.64,aspirational_AP50_95=.70,phase_epochs=3,max_epochs=40,phases=[])
    assert state['target_AP50']==.90 and state['minimum_AP50_95']==.64
    if state.get('status') in {'screen_plateau_below_target','schedule_complete_below_target','storage_limit_below_target'}:
        print('SCREEN_ALREADY_STOPPED',state['status'],flush=True);return
    def save():path.write_text(json.dumps(state,indent=2)+'\n')
    def execute(evaluate_only=False):
        log=RUN/f'phase-{time.time_ns()}.log';state.update(status='running',current_phase_log=str(log));save()
        command=[sys.executable,str(RUN/'run.py')]+(['--evaluate-only'] if evaluate_only else [])
        try:
            with log.open('w') as output:subprocess.run(command,env=dict(os.environ,BOTTLE_PHASE_EPOCHS='3'),stdout=output,stderr=subprocess.STDOUT,check=True)
        except subprocess.CalledProcessError:
            state['status']='failed';save();raise
    reports=policy.verified_records(RUN)
    if any(policy.target_met(r) for r in reports):state['status']='target_verified';save();return
    if (RUN/'progress.json').exists():
        # Finish fresh verification of an interrupted phase before any further update.
        execute(evaluate_only=True)
        reports=policy.verified_records(RUN)
        if any(policy.target_met(r) for r in reports):state['status']='target_verified';save();return
        epoch=load(RUN/'progress.json')['epoch']
        if epoch>state.get('completed_epoch',0):
            outcome=policy.phase_outcome(state,reports,epoch);state.update(outcome);state['phases'].append(outcome);save()
            if state['status']!='running':return
    while True:
        before=load(RUN/'progress.json')['epoch'] if (RUN/'progress.json').exists() else 0
        if before>=40:state['status']='schedule_complete_below_target';save();return
        if shutil.disk_usage(RUN).free<512*1024**2:state['status']='storage_limit_below_target';save();return
        print('START_JOINT_PHASE',before,flush=True);execute()
        epoch=load(RUN/'progress.json')['epoch'];assert before<epoch<=40
        reports=policy.verified_records(RUN)
        outcome=policy.phase_outcome(state,reports,epoch);state.update(outcome);state['phases'].append(outcome);save()
        print('VERIFIED_JOINT_PHASE',json.dumps(outcome),flush=True)
        if state['status']!='running':return

if __name__=='__main__':main()
