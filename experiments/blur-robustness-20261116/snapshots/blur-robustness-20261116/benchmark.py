import json,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent;REPO=R.parents[2];sys.path.insert(0,str(REPO/'scripts'))
from specialists import recognition_at_threshold,sha

def main():
 import torch,cv2
 import live_sorter
 torch.set_num_threads(2)
 f=json.loads((R/'benchmark/freeze.json').read_text());assert sha(f['parent'])==f['parent_sha256'];out=R/'baseline';out.mkdir(exist_ok=False);c=json.loads((REPO/'configs/sorter.json').read_text());model=live_sorter.load_model(f['parent']);pred={}
 for i,r in enumerate(f['records']):
  assert sha(r['image_path'])==r['sha256'];_,dd=live_sorter.predict(model,cv2.imread(r['image_path']),c,'cpu');pred[r['name']]=[[*d.box,d.confidence,d.class_id] for d in dd]
  if i%50==0:print('Processed',i,'of',len(f['records']),flush=True)
 (out/'predictions.json').write_text(json.dumps(pred,indent=2)+'\n');report={}
 for condition in ['clean','gaussian','downsample']:
  rr=[r for r in f['records'] if r['condition']==condition];report[condition]=recognition_at_threshold(rr,{r['name']:pred[r['name']] for r in rr},confidence=.6,minimum_iou=.5);print(condition,report[condition]['micro'],flush=True)
 assert sha(f['parent'])==f['parent_sha256'];(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
