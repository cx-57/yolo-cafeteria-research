"""Fresh source-separated selection, then at most one original benchmark candidate."""
from pathlib import Path
import gc
import json
import torch
from ultralytics import YOLO,__version__
import build
import protocol
import run
RUN=run.RUN;ROOT=run.ROOT;sha=build.sha;save=build.save;load=run.load
OLD=ROOT/'monitor-campaign-v34';LITTER=ROOT/'source-coverage-diagnostic-v40'
COHORTS=[dict(name='openimages',yaml=str(OLD/'data.yaml'),photos=36,boxes=17,fingerprint='1662f8caeec4275e7fdca7c26e89bed6c346227800743fdc20937310e8f0e3d5',dataset_sha=sha(OLD/'dataset.json')),
         dict(name='litter',yaml=str(LITTER/'data.yaml'),photos=32,boxes=18,fingerprint='acf8fd5f6133981c8aae103cf343d9fd04728fc069d23436b5849fbc90b84dd3',dataset_sha=sha(LITTER/'dataset.json'))]
def baseline():
 old=load(OLD/'monitor-initial.json');litter=load(LITTER/'balanced-litter.json')
 assert old['weights_sha']==litter['weights_sha']==run.INITIAL_SHA
 assert old['monitor_fingerprint']==COHORTS[0]['fingerprint'] and litter['cohort_fingerprint']==COHORTS[1]['fingerprint']
 values=[old['AP50'],old['AP50_95'],litter['AP50'],litter['AP50_95']];protocol.valid(values);return values

def evaluate(weights,cohort):
 digest=sha(weights);snapshot=run.WEIGHTS/f'verified-{digest}.pt';build.clone(weights,snapshot)
 path=RUN/f"score-{cohort['name']}-{digest}.json"
 expected=dict(weights_sha=digest,cohort=cohort,dataset_sha=sha(RUN/'dataset.json'),declaration_sha=sha(RUN/'declaration.json'),imgsz=640,batch=4,nms_iou=.5,test_time_augmentation=False,device='mps',ultralytics_version=__version__)
 if path.exists():report=load(path)
 else:
  model=YOLO(str(snapshot));metrics=model.val(data=cohort['yaml'],classes=[0],imgsz=640,batch=4,device='mps',workers=0,augment=False,iou=.5,save_json=True,plots=False,project=str(RUN/'evaluations'),name=f"{cohort['name']}-{digest[:12]}")
  assert list(metrics.box.ap_class_index)==[0] and int(metrics.nt_per_class[0])==cohort['boxes']
  p,r,a,b=metrics.box.class_result(0);report=dict(expected,weights=str(snapshot),AP50=float(a),AP50_95=float(b),precision=float(p),recall=float(r),independent_NHS_test=False)
  assert sha(weights)==sha(snapshot)==digest;save(path,report);del model;gc.collect();torch.mps.empty_cache()
 assert all(report[k]==v for k,v in expected.items())
 protocol.valid([report['AP50'],report['AP50_95'],report['AP50'],report['AP50_95']]);return report

def main():
 run.declaration_check();build.verify();complete=load(RUN/'training-complete.json');assert complete['declaration_sha']==sha(RUN/'declaration.json')
 reference=baseline();candidates=[dict(role='initial',epoch=0,weights_sha=run.INITIAL_SHA,weights=str(run.SOURCE),values=reference)]
 seen={run.INITIAL_SHA}
 for role in ['ap50','localization','latest']:
  proof=load(RUN/(f'native-{role}.json' if role!='latest' else 'progress.json'))
  weights=run.WEIGHTS/f'best-{role}.pt' if role!='latest' else build.CACHE/'train/weights/last.pt'
  digest=sha(weights);assert digest==proof['weights_sha']
  if digest in seen:continue
  seen.add(digest);reports=[evaluate(weights,c) for c in COHORTS];values=[v for report in reports for v in [report['AP50'],report['AP50_95']]]
  candidates.append(dict(role=role,epoch=proof['epoch'],weights=reports[0]['weights'],weights_sha=digest,values=values,reports=reports))
 chosen=max(candidates,key=lambda c:(*protocol.rank(c['values'],reference),c['role']=='initial',-c['epoch']))
 save(RUN/'source-monitor-selection.json',dict(reference=reference,candidates=candidates,chosen=chosen))
 result=dict(chosen=chosen,source_monitor_gate_passed=chosen['role']!='initial' and protocol.eligible(chosen['values'],reference),original_benchmark_used=False,full_training_started=True,declaration_sha=sha(RUN/'declaration.json'))
 if result['source_monitor_gate_passed']:
  locks=build.sources.module('v42_final_lock',ROOT/'monitor-campaign-v34/selection.py');locks.bind_benchmark(RUN/'benchmark-attempt.json',chosen['weights_sha'],sha(RUN/'dataset.json'))
  common=build.sources.module('v42_final_common',ROOT/'plastic-specialist-s-v2/run.py')
  cohort=dict(name='original-benchmark',yaml=str(common.DATA/'data.yaml'),photos=246,boxes=67,fingerprint=common.MANIFEST['source_fingerprint'],validation_protocol=common.MANIFEST['validation_protocol'],dataset_sha=sha(common.DATA/'manifest.json'))
  report=evaluate(Path(chosen['weights']),cohort)
  nonregress=report['AP50']>=.8407106817291071 and report['AP50_95']>=.620605656065828
  result.update(original_benchmark_used=True,benchmark_report=report,promotion_eligible=nonregress,move_on_eligible=nonregress and report['AP50']>=.9,localization_aspiration_met=report['AP50_95']>=.64)
 save(RUN/'campaign-result.json',result);print('FULL_BOTTLE_OUTCOME',json.dumps(result),flush=True)
if __name__=='__main__':main()
