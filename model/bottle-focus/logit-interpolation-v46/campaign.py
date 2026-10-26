"""Two predeclared logit interpolations, source gates, at most one benchmark."""
from pathlib import Path
import argparse
import copy
import gc
import sys
import torch
from ultralytics import YOLO,__version__
from blend import blend_states
RUN=Path(__file__).resolve().parent;ROOT=RUN.parent;PRIOR=ROOT/'hard-negative-logits-v45'
sys.path.insert(0,str(PRIOR))
import data
import run as parent_run
import geometry
import protocol
sys.path.pop(0)
sha=data.sha;load=data.load;save=data.save
CACHE=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-logit-interpolation-v46')
ENDPOINT_SHA='a1789f70f121efe7674b113856d15f178648f2af03e74042b731a6013fa78ccc'
ENDPOINT=data.CACHE/'weights'/f'verified-{ENDPOINT_SHA}.pt'
ALPHAS=[.25,.5]

def frozen_benchmark():
 frozen=load(ROOT/'validation-review/frozen-files-verification.json');benchmark=data.parent.v36.m.DATA
 assert sha(benchmark/'manifest.json')==frozen['manifest_sha']
 assert len(frozen['files'])==246 and sum(r['bottle_boxes'] for r in frozen['files'])==67
 for r in frozen['files']:
  assert sha(benchmark/'images/val'/r['name'])==r['image_sha']
  assert sha(benchmark/'labels/val'/(Path(r['name']).stem+'.txt'))==r['label_sha']
 return frozen

def identity():
 parent_run.check_declaration();audit=data.verify();frozen=frozen_benchmark()
 assert sha(ENDPOINT)==ENDPOINT_SHA
 point=load(PRIOR/'native-ap50.json');assert point['epoch']==1 and point['weights_sha']==ENDPOINT_SHA
 refs=[PRIOR/'declaration.json',PRIOR/'dataset.json',PRIOR/'user-scope.json',PRIOR/'native-ap50.json',PRIOR/'source-monitor-selection.json',ROOT/'validation-review/frozen-files-verification.json',ROOT/'monitor-campaign-v34/monitor-initial.json',ROOT/'source-coverage-diagnostic-v40/balanced-litter.json',ROOT/'monitor-campaign-v34/dataset.json',ROOT/'source-coverage-diagnostic-v40/dataset.json',ROOT/'monitor-campaign-v34/data.yaml',ROOT/'source-coverage-diagnostic-v40/data.yaml']
 return dict(initial_weights_sha=parent_run.INITIAL_SHA,endpoint_weights_sha=ENDPOINT_SHA,alphas=ALPHAS,
   code_shas={p.name:sha(p) for p in sorted(RUN.glob('*.py')) if not p.name.startswith('test_')},
   reference_shas={str(p.relative_to(ROOT)):sha(p) for p in refs},
   dataset_sha=sha(PRIOR/'dataset.json'),parent_code_shas=parent_run.check_declaration()['code_shas'],
   protocol_code_sha=sha(PRIOR/'protocol.py'),geometry_code_sha=sha(PRIOR/'geometry.py'),
   target_scope='disposable_plastic_drink_bottles',maximum_candidates=2,maximum_original_candidates=1,
   gate=dict(retention_tolerance=.005,each_other_source_nonregression=True,mean_other_source_gain=.01),
   benchmark_fingerprint=frozen['source_dataset_fingerprint'],benchmark_protocol=frozen['validation_protocol'],
   promotion_floor=[.8407106817291071,.620605656065828],target_AP50=.9,epochs=0,ultralytics_version=__version__,independent_NHS_test=False)

def verify():
 assert load(RUN/'declaration.json')==identity(),'Frozen experiment identity changed'
 return data.verify()

def build(alpha):
 destination=CACHE/f'alpha-{alpha}.pt';receipt=RUN/f'weights-{alpha}.json'
 if receipt.exists():
  proof=load(receipt);assert proof['alpha']==alpha and proof['declaration_sha']==sha(RUN/'declaration.json') and sha(destination)==proof['weights_sha'];return destination
 left=YOLO(str(parent_run.SOURCE)).model.float().cpu().eval();right=YOLO(str(ENDPOINT)).model.float().cpu().eval()
 allowed=geometry.allowed_names(left);assert allowed==geometry.allowed_names(right) and len(allowed)==6
 reference=geometry.fixed_state(left);geometry.verify_fixed(right,reference)
 result=blend_states(left.state_dict(),right.state_dict(),allowed,alpha);left.load_state_dict(result);geometry.verify_fixed(left,reference)
 checkpoint=torch.load(parent_run.SOURCE,map_location='cpu',weights_only=False)
 checkpoint.update(model=left,ema=None,optimizer=None,epoch=-1,updates=None)
 CACHE.mkdir(parents=True,exist_ok=True);temporary=destination.with_suffix('.partial');torch.save(checkpoint,temporary);temporary.replace(destination)
 reloaded=YOLO(str(destination)).model.float().cpu().eval();geometry.verify_fixed(reloaded,reference)
 assert all(torch.equal(reloaded.state_dict()[k],v) for k,v in result.items())
 proof=dict(alpha=alpha,weights_sha=sha(destination),declaration_sha=sha(RUN/'declaration.json'),mutable_tensors=6,protected_tensor_fingerprint=geometry.fingerprint(reference),roundtrip_exact=True)
 save(receipt,proof);del checkpoint,left,right,reloaded;gc.collect();return destination

def cohorts(audit):
 return [dict(name='retention',yaml=str(PRIOR/'data.yaml'),photos=24,boxes=16,fingerprint=audit['retention_fingerprint'],dataset_sha=sha(PRIOR/'dataset.json'),previously_seen_by_initial=True),
 dict(name='openimages',yaml=str(ROOT/'monitor-campaign-v34/data.yaml'),photos=36,boxes=17,fingerprint='1662f8caeec4275e7fdca7c26e89bed6c346227800743fdc20937310e8f0e3d5',dataset_sha=sha(ROOT/'monitor-campaign-v34/dataset.json')),
 dict(name='litter',yaml=str(ROOT/'source-coverage-diagnostic-v40/data.yaml'),photos=32,boxes=18,fingerprint='acf8fd5f6133981c8aae103cf343d9fd04728fc069d23436b5849fbc90b84dd3',dataset_sha=sha(ROOT/'source-coverage-diagnostic-v40/dataset.json'))]

def evaluate(weights,cohort):
 digest=sha(weights);path=RUN/f"score-{cohort['name']}-{digest}.json"
 expected=dict(weights_sha=digest,cohort=cohort,declaration_sha=sha(RUN/'declaration.json'),imgsz=640,batch=4,nms_iou=.5,test_time_augmentation=False,device='mps',ultralytics_version=__version__)
 if path.exists():return protocol.receipt(load(path),expected)
 model=YOLO(str(weights));assert model.names=={0:'bottle'}
 metrics=model.val(data=cohort['yaml'],classes=[0],imgsz=640,batch=4,device='mps',workers=0,augment=False,iou=.5,save_json=True,plots=False,project=str(RUN/'evaluations'),name=f"{cohort['name']}-{digest[:12]}")
 assert list(metrics.box.ap_class_index)==[0] and int(metrics.nt_per_class[0])==cohort['boxes']
 p,r,a,b=metrics.box.class_result(0);report=dict(expected,weights=str(weights),AP50=float(a),AP50_95=float(b),precision=float(p),recall=float(r),independent_NHS_test=False)
 assert sha(weights)==digest;protocol.receipt(report,expected);save(path,report)
 print('LOGIT_INTERPOLATION_SCORE',cohort['name'],digest,float(a),float(b),flush=True)
 del model;gc.collect();torch.mps.empty_cache();return report

def main(prepare=False):
 torch.set_num_threads(2);torch.mps.set_per_process_memory_fraction(.5)
 if prepare:
  assert not (RUN/'measurement-start.json').exists();save(RUN/'declaration.json',identity());print('INTERPOLATION_PREPARED',sha(RUN/'declaration.json'));return
 audit=verify();locks=data.parent.sources.module('v46_locks',ROOT/'monitor-campaign-v34/selection.py');locks.bind_campaign(RUN/'measurement-start.json',sha(RUN/'declaration.json'))
 reference=load(PRIOR/'source-monitor-selection.json')['reference'];assert len(reference)==6
 candidates=[dict(role='initial',epoch=0,weights=str(parent_run.SOURCE),weights_sha=parent_run.INITIAL_SHA,values=reference)]
 for index,alpha in enumerate(ALPHAS,1):
  path=build(alpha);reports=[evaluate(path,c) for c in cohorts(audit)]
  candidates.append(dict(role=f'alpha-{alpha}',epoch=index,alpha=alpha,weights=str(path),weights_sha=sha(path),values=[v for report in reports for v in [report['AP50'],report['AP50_95']]],reports=reports))
 chosen=protocol.choose(candidates,reference);save(RUN/'source-monitor-selection.json',dict(reference=reference,candidates=candidates,chosen=chosen))
 result=dict(chosen=chosen,retention_and_source_gates_passed=chosen['role']!='initial',original_benchmark_used=False,declaration_sha=sha(RUN/'declaration.json'),independent_NHS_test=False)
 if chosen['role']!='initial':
  frozen_benchmark();locks.bind_benchmark(RUN/'benchmark-attempt.json',chosen['weights_sha'],sha(PRIOR/'dataset.json'))
  common=data.parent.v36.m;manifest=common.MANIFEST
  cohort=dict(name='original-benchmark',yaml=str(common.DATA/'data.yaml'),photos=246,boxes=67,fingerprint=manifest['source_fingerprint'],validation_protocol=manifest['validation_protocol'],dataset_sha=sha(common.DATA/'manifest.json'))
  report=evaluate(Path(chosen['weights']),cohort);nonregress=report['AP50']>=.8407106817291071 and report['AP50_95']>=.620605656065828
  result.update(original_benchmark_used=True,benchmark_report=report,promotion_eligible=nonregress,move_on_eligible=nonregress and report['AP50']>=.9)
 verify();save(RUN/'campaign-result.json',result);print('LOGIT_INTERPOLATION_RESULT',result,flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');main(parser.parse_args().prepare)
