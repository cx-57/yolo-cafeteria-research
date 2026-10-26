"""One fixed distribution-aware quality-weighted two-model WBF candidate."""
from pathlib import Path
import argparse
import gc
import importlib.util
import math
import sys
import torch
from ultralytics import YOLO,__version__
from dual import attach
from quality import install
from fusion import FlipFusionValidator
RUN=Path(__file__).resolve().parent;ROOT=RUN.parent;PRIOR=ROOT/'resolution-monitor-v50'
sys.path.insert(0,str(PRIOR))
spec=importlib.util.spec_from_file_location('v51_prior_campaign',PRIOR/'campaign.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
sys.path.pop(0)
import protocol
sha=prior.sha;load=prior.load;save=prior.save
SOURCE=prior.SOURCE;FIRST_SHA=prior.INITIAL_SHA
SECOND=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-reviewed-full-v42/weights/verified-95370c9b321b57feebbf11204b51a7a8393085dd80bb0ae483b0a1a1f2433013.pt')
SECOND_SHA='95370c9b321b57feebbf11204b51a7a8393085dd80bb0ae483b0a1a1f2433013'
SETTINGS=dict(imgsz=640,batch=4,device='mps',workers=0,augment=True,iou=.5,conf=.001)
FUSION=dict(iou_thr=.55,skip_box_thr=.001,conf_type='avg',weights=[1,1],allows_overflow=False)
PREVIOUS=ROOT/'dual-model-fusion-v51'
FLOORS=load(PREVIOUS/'campaign-result.json')['benchmark_report']
QUALITY=dict(statistic='mean_per_side_top2_DFL_probability_mass',exponent=.5,reg_max=16)

def identity():
 assert load(PRIOR/'declaration.json')==prior.identity()
 assert sha(SOURCE)==FIRST_SHA and sha(SECOND)==SECOND_SHA
 source=load(RUN/'source.json');assert sha(RUN/'vendor_wbf.py')==source['sha256']
 refs=[PRIOR/'declaration.json',PRIOR/'campaign.py',PRIOR/'protocol.py',PRIOR/'dataset.json',ROOT/'reviewed-full-training-v42/campaign-result.json',RUN/'source.json']
 refs += [PREVIOUS/'declaration.json',PREVIOUS/'campaign-result.json']+[PREVIOUS/f"score-{c['name']}.json" for c in prior.cohorts(prior.verify_monitor())]
 return dict(models=[dict(weights=str(SOURCE),sha=FIRST_SHA),dict(weights=str(SECOND),sha=SECOND_SHA)],settings=SETTINGS,fusion=FUSION,quality=QUALITY,code_shas={p.name:sha(p) for p in sorted(RUN.glob('*.py')) if not p.name.startswith('test_')},reference_shas={str(p.relative_to(ROOT)):sha(p) for p in refs},maximum_candidates=1,maximum_original_candidates=1,no_training=True,image_transforms=['identity'],ultralytics_augment_argument_used_for_custom_dual_forward=True,ultralytics_version=__version__,gate=dict(retention_tolerance=.005,other_six_metrics_nonregression=True,other_mean_gain=.01),independent_NHS_test=False)

def controls(cohorts):
 reports=[]
 for c in cohorts:
  report=load(PREVIOUS/f"score-{c['name']}.json")
  protocol.receipt(report,dict(weights_shas=[FIRST_SHA,SECOND_SHA],cohort=c,settings=SETTINGS,fusion=FUSION,declaration_sha=sha(PREVIOUS/'declaration.json'),image_transforms=['identity'],ultralytics_version=__version__))
  reports.append(report)
 return reports

def evaluate(cohort):
 path=RUN/f"score-{cohort['name']}.json";expected=dict(weights_shas=[FIRST_SHA,SECOND_SHA],cohort=cohort,settings=SETTINGS,fusion=FUSION,quality=QUALITY,declaration_sha=sha(RUN/'declaration.json'),image_transforms=['identity'],ultralytics_version=__version__)
 if path.exists():return protocol.receipt(load(path),expected)
 assert sha(SOURCE)==FIRST_SHA and sha(SECOND)==SECOND_SHA
 model=YOLO(str(SOURCE));other=YOLO(str(SECOND)).model
 model.model.fuse(verbose=False);other.fuse(verbose=False);quality_traces=[install(model.model),install(other)];trace=attach(model.model,other);finished={}
 def completed(validator):finished['fused_images']=validator.fused_images
 model.add_callback('on_val_end',completed)
 metrics=model.val(validator=FlipFusionValidator,data=cohort['yaml'],classes=[0],save_json=True,plots=False,project=str(RUN/'evaluations'),name=cohort['name'],**SETTINGS)
 assert finished['fused_images']==cohort['photos'] and trace['calls']==math.ceil(cohort['photos']/SETTINGS['batch'])
 assert sum(s[0] for s in trace['shapes'])==cohort['photos'] and all(t['calls']>=trace['calls'] for t in quality_traces)
 assert list(metrics.box.ap_class_index)==[0] and int(metrics.nt_per_class[0])==cohort['boxes']
 p,r,a,b=metrics.box.class_result(0);report=dict(expected,AP50=float(a),AP50_95=float(b),precision=float(p),recall=float(r),trace=trace,quality_traces=quality_traces,fused_images=finished['fused_images'],speed_ms_per_image=metrics.speed,independent_NHS_test=False)
 assert sha(SOURCE)==FIRST_SHA and sha(SECOND)==SECOND_SHA;protocol.receipt(report,expected);save(path,report);print('DUAL_MODEL_SCORE',cohort['name'],float(a),float(b),flush=True)
 del model,other;gc.collect();torch.mps.empty_cache();return report

def main(prepare=False):
 torch.set_num_threads(2);torch.mps.set_per_process_memory_fraction(.5)
 if prepare:
  assert not (RUN/'measurement-start.json').exists();save(RUN/'declaration.json',identity());print('DUAL_MODEL_PREFLIGHT',sha(RUN/'declaration.json'));return
 assert load(RUN/'declaration.json')==identity();protocol.bind_attempt(RUN/'measurement-start.json',dict(declaration_sha=sha(RUN/'declaration.json')))
 cohorts=prior.cohorts(prior.verify_monitor());baseline=controls(cohorts);reports=[evaluate(c) for c in cohorts]
 initial=[v for r in baseline for v in [r['AP50'],r['AP50_95']]];candidate=[v for r in reports for v in [r['AP50'],r['AP50_95']]];accepted=protocol.eligible(candidate,initial)
 save(RUN/'source-comparison.json',dict(controls=baseline,candidate_reports=reports,baseline_values=initial,candidate_values=candidate,gate_passed=accepted))
 result=dict(weights_shas=[FIRST_SHA,SECOND_SHA],ensemble=True,source_gates_passed=accepted,original_benchmark_used=False,declaration_sha=sha(RUN/'declaration.json'),independent_NHS_test=False)
 if accepted:
  proof=prior.frozen();protocol.bind_attempt(RUN/'benchmark-attempt.json',dict(weights_shas=[FIRST_SHA,SECOND_SHA],settings=SETTINGS,fusion=FUSION,quality=QUALITY,declaration_sha=sha(RUN/'declaration.json'),manifest_sha=proof['manifest_sha']))
  common=prior.data.parent.v36.m;cohort=dict(name='original-benchmark',yaml=str(common.DATA/'data.yaml'),photos=246,boxes=67,fingerprint=proof['source_dataset_fingerprint'],parent_validation_protocol=proof['validation_protocol'],dataset_sha=sha(common.DATA/'manifest.json'))
  report=evaluate(cohort);good=report['AP50']>=FLOORS['AP50'] and report['AP50_95']>=FLOORS['AP50_95']
  result.update(original_benchmark_used=True,benchmark_report=report,promotion_eligible=good,user_aspiration_met=good and report['AP50']>=.86 and report['AP50_95']>=.65)
 assert load(RUN/'declaration.json')==identity();save(RUN/'campaign-result.json',result);print('DUAL_MODEL_OUTCOME',result,flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');main(parser.parse_args().prepare)
