"""One preregistered inference-resolution alternative; no weight or cutoff changes."""
import json,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent;REPO=R.parents[2];sys.path.insert(0,str(REPO/'scripts'))
from specialists import sha,recognition_at_threshold,score_predictions,NAMES

def main():
 import cv2,torch,live_sorter
 torch.set_num_threads(2);f=json.loads((R/'benchmark/freeze.json').read_text());p=json.loads((R/'resolution-protocol.json').read_text());assert sha(f['parent'])==f['parent_sha256'];out=R/'resolution960';out.mkdir(exist_ok=False);c=json.loads((REPO/'configs/sorter.json').read_text());c['imgsz']=p['imgsz'];model=live_sorter.load_model(f['parent']);pred={};start=time.monotonic()
 for i,r in enumerate(f['records']):
  assert sha(r['image_path'])==r['sha256'];_,dd=live_sorter.predict(model,cv2.imread(r['image_path']),c,'cpu');pred[r['name']]=[[*d.box,d.confidence,d.class_id] for d in dd]
  if i%100==0:print('PROGRESS',i,len(f['records']),flush=True)
 elapsed=time.monotonic()-start;baseline=json.loads((R/'baseline/predictions.json').read_text());result={};checks={}
 for condition in ['clean','gaussian','downsample']:
  rr=[r for r in f['records'] if r['condition']==condition];results={}
  for name,pp in [('before',baseline),('after',pred)]:
   subset={r['name']:pp[r['name']] for r in rr};results[name]=dict(operating=recognition_at_threshold(rr,subset,confidence=.6,minimum_iou=.5),ap_at_occupancy_floor=score_predictions(rr,subset,set(range(7))))
  b=results['before']['operating']['micro'];a=results['after']['operating']['micro'];checks[condition]=dict(recall_delta=a['recall']-b['recall'],precision_delta=a['precision']-b['precision'],accept=a['precision']>=b['precision']-.02 and a['recall']>=b['recall']+(.02 if condition!='clean' else -.02));result[condition]=results;print('RESULT',condition,b,a,flush=True)
 checks['clean_class_retention']=all(result['clean']['after']['ap_at_occupancy_floor']['per_class'][n]['AP50']>=result['clean']['before']['ap_at_occupancy_floor']['per_class'][n]['AP50']-.02 for n in NAMES)
 report=dict(results=result,checks=checks,accept=all(v['accept'] for k,v in checks.items() if k!='clean_class_retention') and checks['clean_class_retention'],seconds=elapsed,seconds_per_image=elapsed/len(pred),parent_sha256=sha(f['parent']),protocol_sha256=sha(R/'resolution-protocol.json'),ap_input_confidence_floor=.15,real_camera_certified=False)
 (out/'predictions.json').write_text(json.dumps(pred,indent=2)+'\n');(out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print('ACCEPT',report['accept'],checks,flush=True)
if __name__=='__main__':main()
