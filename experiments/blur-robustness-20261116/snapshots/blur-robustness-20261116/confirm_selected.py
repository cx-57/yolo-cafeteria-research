"""One-shot confirmation; requires all frozen development acceptance gates."""
import json,sys
from pathlib import Path
R=Path(__file__).resolve().parent; REPO=R.parents[2]
sys.path.insert(0,str(REPO/'scripts'))
from specialists import sha,recognition_at_threshold,score_predictions,NAMES

from retention import class_operating_retention

def main():
 comparison=json.loads((R/'comparison.json').read_text())
 if not comparison['accept']:
  raise SystemExit('Development gates failed. Confirmation remains unopened.')
 protocol=json.loads((R/'trial-protocol.json').read_text())
 assert sha(R/'trial-protocol.json')==comparison['protocol_sha256']
 assert sha(R/'class-retention-amendment.json')==comparison['class_retention_amendment_sha256']
 frozen=R/'confirmation/freeze.json'; freeze=json.loads(frozen.read_text())
 checkpoints={'parent':{'checkpoint':protocol['parent'],'sha256':protocol['parent_sha256']},**comparison['proof']}
 selection={'development_comparison_sha256':sha(R/'comparison.json'),'confirmation_sha256':sha(frozen),'checkpoints':checkpoints,'inference_config_sha256':comparison['inference_config_sha256'],'selected':'candidate','confidence':.60,'minimum_iou':.50,'imgsz':640,'automatic_promotion':False}
 out=R/'confirmation-evaluation';out.mkdir(exist_ok=False)
 # Selection is durable before any confirmation predictions exist.
 (out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
 import torch,cv2,live_sorter
 torch.set_num_threads(2)
 c=json.loads((R/'inference-config.json').read_text());assert sha(R/'inference-config.json')==comparison['inference_config_sha256'];assert c['confidence']==.60 and c['imgsz']==640
 results={}
 for name,checkpoint in checkpoints.items():
  assert sha(checkpoint['checkpoint'])==checkpoint['sha256']
  model=live_sorter.load_model(checkpoint['checkpoint']);pred={}
  for i,r in enumerate(freeze['records']):
   assert sha(r['image_path'])==r['sha256']
   _,dd=live_sorter.predict(model,cv2.imread(r['image_path']),c,'cpu')
   pred[r['name']]=[[*d.box,d.confidence,d.class_id] for d in dd]
   if i%100==0:print(name,i,'/',len(freeze['records']),flush=True)
  assert sha(checkpoint['checkpoint'])==checkpoint['sha256']
  (out/f'{name}-predictions.json').write_text(json.dumps(pred,indent=2)+'\n');results[name]={}
  for condition in ['clean','gaussian','downsample']:
   rr=[r for r in freeze['records'] if r['condition']==condition];pp={r['name']:pred[r['name']] for r in rr}
   results[name][condition]={'operating':recognition_at_threshold(rr,pp,.60,.50),'ap':score_predictions(rr,pp,set(range(7)))}
   print(name,condition,results[name][condition]['operating']['micro'],flush=True)
 checks={}
 for reference in ['parent','control']:
  checks[reference]={}
  for condition in ['clean','gaussian','downsample']:
   before=results[reference][condition]['operating']['micro'];after=results['candidate'][condition]['operating']['micro']
   checks[reference][condition]={'recall_delta':after['recall']-before['recall'],'precision_delta':after['precision']-before['precision'],'acceptable':after['recall']>=before['recall']+(-.02 if condition=='clean' else .02) and after['precision']>=before['precision']-.02}
  checks[reference]['class_retention']=all(results['candidate']['clean']['ap']['per_class'][n]['AP50']>=results[reference]['clean']['ap']['per_class'][n]['AP50']-.02 for n in NAMES)
 passed=all(v['acceptable'] for d in checks.values() for k,v in d.items() if k!='class_retention') and all(d['class_retention'] for d in checks.values())
 report={'results':results,'checks':checks,'accept':passed,'selection_sha256':sha(out/'selection.json'),'ap_input_confidence_floor':.15,'scope':freeze['scope'],'automatic_promotion':False}
 report['original_gates_accept']=report['accept']
 report['class_operating_retention']={ref:class_operating_retention(results,ref) for ref in ['parent','control']}
 report['accept']=report['accept'] and all(v['acceptable'] for d in report['class_operating_retention'].values() for cc in d.values() for v in cc.values())
 report['class_retention_amendment_sha256']=sha(R/'class-retention-amendment.json')
 assert sha(frozen)==selection['confirmation_sha256']
 (out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print('CONFIRMATION ACCEPT',report['accept'],flush=True)
if __name__=='__main__':main()
