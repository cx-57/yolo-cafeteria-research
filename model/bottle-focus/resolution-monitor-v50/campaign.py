"""Fixed incumbent, one896ordinary candidate, matched640development controls."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import shutil
import sys
import torch
import yaml
from ultralytics import YOLO,__version__
import protocol
RUN=Path(__file__).resolve().parent;ROOT=RUN.parent;PREV=ROOT/'box-projection-v48';STAGE=ROOT/'taco-drink-staging-v49'
sys.path.insert(0,str(PREV))
import data
import run as prior_run
sys.path.pop(0)
sha=data.sha;load=data.load;save=data.save
CACHE=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-resolution-monitor-v50')
SOURCE=prior_run.SOURCE;INITIAL_SHA=prior_run.INITIAL_SHA
SETTINGS=dict(batch=4,device='mps',workers=0,augment=False,iou=.5,conf=.001)
CONTROL=640;CANDIDATE=896

def derive_monitor():
 photos=load(STAGE/'downloads.json');first=load(STAGE/'first-review.json');second=load(STAGE/'second-review-independent.json');follow=load(STAGE/'followup-review.json');audit=load(STAGE/'geometry-audit.json');screen=load(STAGE/'overlap-screen.json')
 assert first['source_downloads_sha']==second['source_downloads_sha']==follow['source_downloads_sha']==audit['source_downloads_sha']==sha(STAGE/'downloads.json')
 assert follow['prior_independent_review_sha']==sha(STAGE/'second-review-independent.json') and audit['audit_code_sha']==sha(STAGE/'geometry_audit.py')
 assert follow['evidence_shas']=={n:sha(STAGE/n) for n in follow['evidence_shas']}
 data.parent.sources.helper.check_inputs(screen['inputs']);old=data.verify();prior=load(ROOT/'classifier-retention-v44/source-selection.json')
 blocked=old['training_records']+old['retention_records']+prior['old_monitor_records'];groups={data.prior.weights.canonical_group(r) for r in blocked};bytes_seen={r['image_sha'] for r in blocked};pixels_seen={r['pixels'] for r in blocked};records=[];held=[]
 for item in screen['records']:
  i=item['index'];row=photos[i];a=first['records'][i];b=second['records'][i]
  assert a['index']==b['index']==i and a['id']==b['id']==row['id']==item['id']
  assert a['sha']==b['image_sha']==row['sha']==item['sha']
  assert a['decision']==b['decision']=='candidate' and i in follow['revised_independent_candidate_indices'] and audit['records'][i]['accepted_geometry'] is True
  if item['hold']:continue
  assert all(not item[k] for k in ['byte_matches','pixel_matches','dhash_matches','feature_matches'])
  group='flickr/'+row['photo_metadata']['photo_id']
  if group in groups:held.append(row['id']);continue
  assert row['sha'] not in bytes_seen and row['pixels'] not in pixels_seen
  assert row['photo_metadata']['license_url'].startswith('https://creativecommons.org/licenses/') and row['status']=='unreviewed_staged_only'
  width,height=row['original']['width'],row['original']['height'];labels=[]
  for target in row['annotations']:
   assert target['category_id']==5;x,y,w,h=target['bbox'];assert 0<=x<x+w<=width and 0<=y<y+h<=height
   labels.append('0 '+' '.join(f'{v:.9f}' for v in [(x+w/2)/width,(y+h/2)/height,w/width,h/height]))
  image=CACHE/'images/val'/Path(row['image']).name;label=CACHE/'labels/val'/(image.stem+'.txt')
  records.append(dict(id=row['id'],source_index=i,source_image=row['image'],image=str(image),label=str(label),image_sha=row['sha'],pixels=row['pixels'],source_group=group,rows=labels,previously_trained=False,human_verified=False))
  groups.add(group);bytes_seen.add(row['sha']);pixels_seen.add(row['pixels'])
 return records,held

def make_monitor():
 records,held=derive_monitor();assert len(records)>=4 and sum(len(r['rows']) for r in records)>=4,'Insufficient fresh monitor support'
 for r in records:
  image=Path(r['image']);label=Path(r['label']);image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
  if not image.exists():shutil.copyfile(r['source_image'],image)
  if label.exists():assert label.read_text().splitlines()==r['rows']
  else:label.write_text('\n'.join(r['rows'])+'\n')
 config=dict(path=str(CACHE),train='images/val',val='images/val',names=['bottle']);(RUN/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
 fingerprint=hashlib.sha256(json.dumps([{k:r[k] for k in ['id','image_sha','pixels','source_group','rows']} for r in records],sort_keys=True).encode()).hexdigest()
 save(RUN/'dataset.json',dict(monitor_records=records,source_group_holds=held,monitor_photos=len(records),monitor_boxes=sum(len(r['rows']) for r in records),monitor_fingerprint=fingerprint,config=config,permanent_training_exclusion=True,no_training=True,positive_only_monitor=True,independent_NHS_test=False))

def verify_monitor():
 audit=load(RUN/'dataset.json');records,held=derive_monitor();assert audit['monitor_records']==records and audit['source_group_holds']==held
 assert yaml.safe_load((RUN/'data.yaml').read_text())==audit['config']
 expected_fingerprint=hashlib.sha256(json.dumps([{k:r[k] for k in ['id','image_sha','pixels','source_group','rows']} for r in records],sort_keys=True).encode()).hexdigest()
 assert audit['monitor_fingerprint']==expected_fingerprint and audit['monitor_photos']==len(records)>=4 and audit['monitor_boxes']==sum(len(r['rows']) for r in records)>=4
 actual={str(p) for p in (CACHE/'images/val').iterdir()};assert actual=={r['image'] for r in records},'Unexpected monitor image'
 for r in records:
  assert sha(r['image'])==r['image_sha'] and Path(r['label']).read_text().splitlines()==r['rows']
 return audit

def frozen():
 proof=load(ROOT/'validation-review/frozen-files-verification.json');benchmark=data.parent.v36.m.DATA
 assert sha(benchmark/'manifest.json')==proof['manifest_sha']
 for r in proof['files']:
  assert sha(benchmark/'images/val'/r['name'])==r['image_sha'] and sha(benchmark/'labels/val'/(Path(r['name']).stem+'.txt'))==r['label_sha']
 return proof

def identity():
 prior_run.check_declaration();assert sha(SOURCE)==INITIAL_SHA;frozen();audit=verify_monitor()
 prior=load(PREV/'declaration.json');assert prior['reference_shas']=={n:sha(ROOT/n) for n in prior['reference_shas']}
 refs=[PREV/'declaration.json',PREV/'dataset.json',RUN/'dataset.json',RUN/'data.yaml',ROOT/'validation-review/frozen-files-verification.json']+[STAGE/n for n in ['source-selection.json','downloads.json','first-review.json','second-review-independent.json','followup-review.json','geometry-audit.json','geometry_audit.py','overlap-screen.json','screen.py']]
 return dict(weights_sha=INITIAL_SHA,control_imgsz=CONTROL,candidate_imgsz=CANDIDATE,settings=SETTINGS,code_shas={p.name:sha(p) for p in sorted(RUN.glob('*.py')) if not p.name.startswith('test_')},reference_shas={str(p.relative_to(ROOT)):sha(p) for p in refs},parent_code_shas=prior['code_shas'],ultralytics_version=__version__,fresh_monitor_fingerprint=audit['monitor_fingerprint'],maximum_candidates=1,maximum_original_candidates=1,epochs=0,gate=dict(retention_tolerance=.005,other_six_metrics_nonregression=True,other_mean_gain=.01),original_AP_floors=[.8407106817291071,.620605656065828],independent_NHS_test=False)

def cohorts(audit):
 return [dict(name='retention',yaml=str(PREV/'data.yaml'),photos=24,boxes=16,fingerprint=load(PREV/'dataset.json')['retention_fingerprint'],dataset_sha=sha(PREV/'dataset.json'),previously_seen_by_initial=True),
 dict(name='openimages',yaml=str(ROOT/'monitor-campaign-v34/data.yaml'),photos=36,boxes=17,dataset_sha=sha(ROOT/'monitor-campaign-v34/dataset.json')),
 dict(name='litter',yaml=str(ROOT/'source-coverage-diagnostic-v40/data.yaml'),photos=32,boxes=18,dataset_sha=sha(ROOT/'source-coverage-diagnostic-v40/dataset.json')),
 dict(name='fresh-litter',yaml=str(RUN/'data.yaml'),photos=audit['monitor_photos'],boxes=audit['monitor_boxes'],fingerprint=audit['monitor_fingerprint'],dataset_sha=sha(RUN/'dataset.json'),previously_trained=False,positive_only=True)]

def evaluate(cohort,imgsz):
 path=RUN/f"score-{cohort['name']}-{imgsz}.json";expected=dict(weights_sha=INITIAL_SHA,cohort=cohort,imgsz=imgsz,settings=SETTINGS,declaration_sha=sha(RUN/'declaration.json'),ultralytics_version=__version__)
 if path.exists():return protocol.receipt(load(path),expected)
 assert sha(SOURCE)==INITIAL_SHA;model=YOLO(str(SOURCE));assert model.names=={0:'bottle'}
 metrics=model.val(data=cohort['yaml'],classes=[0],imgsz=imgsz,save_json=True,plots=False,project=str(RUN/'evaluations'),name=f"{cohort['name']}-{imgsz}",**SETTINGS)
 assert list(metrics.box.ap_class_index)==[0] and int(metrics.nt_per_class[0])==cohort['boxes']
 p,r,a,b=metrics.box.class_result(0);report=dict(expected,weights=str(SOURCE),AP50=float(a),AP50_95=float(b),precision=float(p),recall=float(r),speed_ms_per_image=metrics.speed,independent_NHS_test=False)
 assert sha(SOURCE)==INITIAL_SHA;protocol.receipt(report,expected);save(path,report);print('RESOLUTION_SCORE',cohort['name'],imgsz,float(a),float(b),flush=True)
 del model;gc.collect();torch.mps.empty_cache();return report

def main(prepare=False):
 torch.set_num_threads(2);torch.mps.set_per_process_memory_fraction(.5)
 if prepare:
  assert not (RUN/'measurement-start.json').exists();make_monitor();save(RUN/'declaration.json',identity());print('RESOLUTION_PREFLIGHT',sha(RUN/'declaration.json'));return
 assert load(RUN/'declaration.json')==identity();protocol.bind_attempt(RUN/'measurement-start.json',dict(declaration_sha=sha(RUN/'declaration.json')));audit=verify_monitor()
 controls=[evaluate(c,CONTROL) for c in cohorts(audit)];candidates=[evaluate(c,CANDIDATE) for c in cohorts(audit)]
 baseline=[v for r in controls for v in [r['AP50'],r['AP50_95']]];candidate=[v for r in candidates for v in [r['AP50'],r['AP50_95']]];accepted=protocol.eligible(candidate,baseline)
 save(RUN/'source-comparison.json',dict(controls=controls,candidates=candidates,baseline_values=baseline,candidate_values=candidate,gate_passed=accepted))
 result=dict(weights_sha=INITIAL_SHA,imgsz=CANDIDATE,source_gates_passed=accepted,original_benchmark_used=False,declaration_sha=sha(RUN/'declaration.json'),independent_NHS_test=False)
 if accepted:
  proof=frozen();protocol.bind_attempt(RUN/'benchmark-attempt.json',dict(weights_sha=INITIAL_SHA,imgsz=CANDIDATE,settings=SETTINGS,declaration_sha=sha(RUN/'declaration.json'),manifest_sha=proof['manifest_sha']))
  common=data.parent.v36.m;cohort=dict(name='original-benchmark',yaml=str(common.DATA/'data.yaml'),photos=246,boxes=67,fingerprint=proof['source_dataset_fingerprint'],parent_validation_protocol=proof['validation_protocol'],dataset_sha=sha(common.DATA/'manifest.json'))
  report=evaluate(cohort,CANDIDATE);good=report['AP50']>=.8407106817291071 and report['AP50_95']>=.620605656065828
  result.update(original_benchmark_used=True,benchmark_report=report,promotion_eligible=good,move_on_eligible=good and report['AP50']>=.9,inference_protocol_changed=True)
 assert load(RUN/'declaration.json')==identity();save(RUN/'campaign-result.json',result);print('RESOLUTION_OUTCOME',result,flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');main(parser.parse_args().prepare)
