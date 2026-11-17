"""Fixed detection-score consensus, with no threshold search or new data selection."""
import json,math,sys
from pathlib import Path
R=Path(__file__).resolve().parent;REPO=R.parents[2];sys.path.insert(0,str(REPO/'scripts'))
from specialists import sha,recognition_at_threshold,score_predictions,NAMES
from evaluate_sorting import iou
from retention import class_operating_retention

def combine(parent,adapted):
 result=[]
 for p in parent:
  # Keep original box and class; candidate must agree at .60 and IoU .50.
  matches=[c for c in adapted if c[5]==p[5] and c[4]>=.60 and iou(p[:4],c[:4])>=.50]
  evidence=max([math.sqrt(p[4]*c[4]) for c in matches],default=0.)
  result.append([*p[:4],max(p[4],evidence),p[5]])
 return result

def main():
 out=R.parent/'blur-consensus-20261116';out.mkdir(exist_ok=False)
 original=json.loads((R/'comparison.json').read_text());assert not original['accept']
 source={name:R/f'evaluation-{name}/predictions.json' for name in ['parent','control','candidate']}
 protocol={'hypothesis':'Conservative agreement-based score fusion: preserve original detections/boxes, and only boost a borderline original detection when trained model agrees on class and IoU>=.50 with confidence>=.60. Fused confidence max(parent,sqrt(parent*candidate)); acceptance remains .60. Same transformation for control and candidate. No parameter search.','source_predictions':{n:{'path':str(p),'sha256':sha(p)} for n,p in source.items()},'benchmark_sha256':sha(R/'benchmark/freeze.json'),'config_sha256':sha(R/'inference-config.json'),'development_comparison_sha256':sha(R/'comparison.json'),'class_amendment_sha256':sha(R/'class-retention-amendment.json'),'fixed_confidence':.60,'matching_iou':.50,'confirmation_opened':False,'automatic_promotion':False}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 pred={n:json.loads(p.read_text()) for n,p in source.items()};allpred={'parent':pred['parent']}
 for name in ['control','candidate']:
  allpred[name]={k:combine(rows,pred[name][k]) for k,rows in pred['parent'].items()}
  (out/f'{name}-predictions.json').write_text(json.dumps(allpred[name],indent=2)+'\n')
 records=json.loads((R/'benchmark/freeze.json').read_text())['records'];results={}
 for name,prediction in allpred.items():
  results[name]={}
  for condition in ['clean','gaussian','downsample']:
   rr=[r for r in records if r['condition']==condition];pp={r['name']:prediction[r['name']] for r in rr}
   results[name][condition]={'operating':recognition_at_threshold(rr,pp,.60,.50),'ap':score_predictions(rr,pp,set(range(7)))}
 checks={}
 for ref in ['parent','control']:
  checks[ref]={}
  for condition in ['clean','gaussian','downsample']:
   b=results[ref][condition]['operating']['micro'];a=results['candidate'][condition]['operating']['micro']
   checks[ref][condition]={'recall_delta':a['recall']-b['recall'],'precision_delta':a['precision']-b['precision'],'acceptable':a['precision']>=b['precision']-.02 and a['recall']>=b['recall']+(-.02 if condition=='clean' else .02)}
  checks[ref]['class_retention']=all(results['candidate']['clean']['ap']['per_class'][n]['AP50']>=results[ref]['clean']['ap']['per_class'][n]['AP50']-.02 for n in NAMES)
 original_accept=all(v['acceptable'] for d in checks.values() for k,v in d.items() if k!='class_retention') and all(d['class_retention'] for d in checks.values())
 class_checks={ref:class_operating_retention(results,ref) for ref in ['parent','control']};accept=original_accept and all(v['acceptable'] for d in class_checks.values() for cc in d.values() for v in cc.values())
 report={'results':results,'checks':checks,'original_gates_accept':original_accept,'class_operating_retention':class_checks,'accept':accept,'protocol_sha256':sha(out/'protocol.json'),'ap_input_confidence_floor':.15,'scope':'Exposed synthetic development benchmark; original detections retained, consensus may raise borderline scores; not real-camera certification.'}
 (out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
 print('ACCEPT',accept,'original gates',original_accept,checks,flush=True)
 for name,d in results.items():print(name,{c:v['operating']['micro'] for c,v in d.items()},flush=True)
 print('CLASS FAILURES',[(ref,c,n) for ref,d in class_checks.items() for c,cc in d.items() for n,v in cc.items() if not v['acceptable']],flush=True)
if __name__=='__main__':main()
