"""Compare matched checkpoints at the frozen operating point; never tune screenshots."""
import json,sys
from pathlib import Path
R=Path(__file__).resolve().parent;REPO=R.parents[2];A=REPO.parent/'yolo-cafeteria-research-assets';sys.path.insert(0,str(REPO/'scripts'))
from specialists import sha,recognition_at_threshold,score_predictions,NAMES

from retention import class_operating_retention

def main():
 import cv2,torch,live_sorter
 torch.set_num_threads(2);f=json.loads((R/'benchmark/freeze.json').read_text());protocol=json.loads((R/'trial-protocol.json').read_text());assert sha(R/'benchmark/freeze.json')==protocol['benchmark_sha256'];c=json.loads((R/'inference-config.json').read_text());assert c['confidence']==.60 and c['occupancy_confidence']==.15 and c['imgsz']==640;records=f['records'];allpred={};proof={}
 for name in ['parent','control','candidate']:
  if name=='parent':
   checkpoint=Path(protocol['parent']);assert sha(checkpoint)==protocol['parent_sha256'];recipe_digest=None
  else:
   folder=A/f'artifacts/blur-neck-{name}-20261116';status=json.loads((folder/'status.json').read_text());assert status['state']=='complete';recipe=json.loads((folder/'recipe.json').read_text());assert recipe['model_sha256']==protocol['parent_sha256'] and recipe['applied_settings']['epochs']==2 and recipe['applied_settings']['freeze']==protocol['freeze_layers']
   checkpoint=folder/'weights/last.pt';recipe_digest=sha(folder/'recipe.json')
  digest=sha(checkpoint);out=R/f'evaluation-{name}';out.mkdir(exist_ok=False);model=live_sorter.load_model(str(checkpoint));pred={}
  for i,r in enumerate(records):
   assert sha(r['image_path'])==r['sha256'];_,dd=live_sorter.predict(model,cv2.imread(r['image_path']),c,'cpu');pred[r['name']]=[[*d.box,d.confidence,d.class_id] for d in dd]
   if i%100==0:print(name,i,'of',len(records),flush=True)
  assert sha(checkpoint)==digest;(out/'predictions.json').write_text(json.dumps(pred,indent=2)+'\n');allpred[name]=pred;proof[name]=dict(checkpoint=str(checkpoint),sha256=digest,recipe_sha256=recipe_digest)
 results={}
 for name,pred in allpred.items():
  results[name]={}
  for condition in ['clean','gaussian','downsample']:
   rr=[r for r in records if r['condition']==condition];pp={r['name']:pred[r['name']] for r in rr};results[name][condition]=dict(operating=recognition_at_threshold(rr,pp,confidence=.6,minimum_iou=.5),ap=score_predictions(rr,pp,set(range(7))));print('RESULT',name,condition,results[name][condition]['operating']['micro'],flush=True)
 checks={}
 for reference in ['parent','control']:
  checks[reference]={}
  for condition in ['clean','gaussian','downsample']:
   before=results[reference][condition]['operating']['micro'];after=results['candidate'][condition]['operating']['micro'];checks[reference][condition]=dict(recall_delta=after['recall']-before['recall'],precision_delta=after['precision']-before['precision'],acceptable=after['precision']>=before['precision']-.02 and after['recall']>=before['recall']+(.02 if condition!='clean' else -.02))
  checks[reference]['class_retention']=all(results['candidate']['clean']['ap']['per_class'][n]['AP50']>=results[reference]['clean']['ap']['per_class'][n]['AP50']-.02 for n in NAMES)
 accept=all(v['acceptable'] for d in checks.values() for k,v in d.items() if k!='class_retention') and all(d['class_retention'] for d in checks.values())
 report=dict(inference_config_sha256=sha(R/'inference-config.json'),inference_config=c,results=results,checks=checks,accept=accept,proof=proof,protocol_sha256=sha(R/'trial-protocol.json'),benchmark_sha256=sha(R/'benchmark/freeze.json'),real_camera_certified=False,ap_input_confidence_floor=.15,ap_scope='AP computed from the identical occupancy-level prediction floor; operating precision/recall at0.60 are the primary measurements.')
 report['original_gates_accept']=report['accept']
 report['class_operating_retention']={ref:class_operating_retention(results,ref) for ref in ['parent','control']}
 report['accept']=report['accept'] and all(v['acceptable'] for d in report['class_operating_retention'].values() for cc in d.values() for v in cc.values())
 report['class_retention_amendment_sha256']=sha(R/'class-retention-amendment.json')
 (R/'comparison.json').write_text(json.dumps(report,indent=2)+'\n');print('ACCEPT',report['accept'],checks,flush=True)
if __name__=='__main__':main()
