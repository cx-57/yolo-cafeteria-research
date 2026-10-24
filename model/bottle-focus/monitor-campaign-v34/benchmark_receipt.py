"""Recover fully saved benchmark reports without inference or candidate reselection."""
from pathlib import Path
import hashlib
import json
import shutil


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.partial');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)

def reconcile(run,report_path,chosen_weights,manifest,policy):
 run=Path(run);report=json.loads(Path(report_path).read_text());digest=sha(chosen_weights);policy.metrics(report)
 expected=dict(weights_sha=digest,role='monitor-selected',completed_epoch=9,imgsz=640,batch=4,nms_iou=.5,test_time_augmentation=False,device='mps',validation_images=246,bottle_objects=67,training_dataset_audit_sha=sha(run/'dataset.json'),source_dataset_fingerprint=manifest['source_fingerprint'],validation_protocol=manifest['validation_protocol'],ultralytics_version='8.4.175',independent_test=False)
 assert all(report.get(k)==v for k,v in expected.items()),'Saved benchmark receipt does not match bound candidate/protocol'
 assert Path(report['source_checkpoint']).resolve()==Path(chosen_weights).resolve()
 assert Path(report['weights']).resolve()==(run/f'verified-{digest}.pt').resolve() and sha(report['weights'])==digest
 history=policy.verified_records(run);same=[r for r in history if r['weights_sha']==digest]
 assert not any(r!=report for r in same),'Conflicting saved benchmark receipts'
 if not same:history.append(report)
 save(run/'metrics-history.json',history)
 save(run/'frontier.json',dict(target_AP50=.90,minimum_AP50_95=.64,aspirational_AP50_95=.70,members=policy.frontier(history)))
 return report

def promote(run,report_path,report,runtime_copy):
 run=Path(run);incumbent=run.parent/'verified-incumbent';proof_path=incumbent/'verification.json';old=json.loads(proof_path.read_text())
 if not (report['AP50']>=old['AP50'] and report['AP50_95']>=old['AP50_95'] and (report['AP50']>old['AP50'] or report['AP50_95']>old['AP50_95'])):return False
 assert json.loads(Path(report_path).read_text())==report and sha(report['weights'])==report['weights_sha']
 for destination in (incumbent/'bottle.pt',Path(runtime_copy)):
  if destination.exists():assert sha(destination) in {old['weights_sha'],report['weights_sha']},'Unexpected incumbent bytes during recovery'
  destination.parent.mkdir(parents=True,exist_ok=True);temporary=destination.with_suffix('.pt.partial');shutil.copyfile(report['weights'],temporary);assert sha(temporary)==report['weights_sha'];temporary.replace(destination)
 save(proof_path,dict(report,weights=str(incumbent/'bottle.pt'),source_verification=str(report_path),runtime_cache_copy=str(runtime_copy)))
 return True
