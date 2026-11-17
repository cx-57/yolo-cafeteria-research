"""Diagnostic-only user screenshots. Never training/threshold/model selection."""
import json,sys
from pathlib import Path
R=Path(__file__).resolve().parent;REPO=R.parents[2];sys.path.insert(0,str(REPO/'scripts'))
from specialists import sha,NAMES

def main():
 comparison=json.loads((R/'comparison.json').read_text()) # Explicit user request: diagnostic-only comparison, regardless of acceptance.
 import cv2,torch,live_sorter
 torch.set_num_threads(1)
 config=json.loads((R/'inference-config.json').read_text());assert sha(R/'inference-config.json')==comparison['inference_config_sha256']
 out=R/'screenshot-diagnostics';out.mkdir(exist_ok=False)
 cases={'8.30.37':'food-like bread','8.36.52':'wrapper (explicit user correction)','8.38.51':'wrapper (same objects, another view)','8.40.49':None,'8.39.53':'food-like bread'}
 records=[]
 for timestamp,truth in cases.items():
  paths=list((Path.home()/'Desktop').glob(f'Screenshot 2026-11-16 at {timestamp}*PM.png'));assert len(paths)==1
  p=paths[0];records.append({'path':str(p),'sha256':sha(p),'time':timestamp,'description':truth,'ground_truth_boxes':None})
 result={'development_accepted':comparison['accept'],'scope':'Diagnostic only. Correlated screenshots are not independent photo samples; unknown 8.40.49 ground truth remains unknown. No box annotations, no detection-accuracy estimate.','records':records,'models':{},'confidence':.60,'selection_comparison_sha256':sha(R/'comparison.json')}
 for name,proof in comparison['proof'].items():
  assert sha(proof['checkpoint'])==proof['sha256'];model=live_sorter.load_model(proof['checkpoint']);predictions={}
  for record in records:
   assert sha(record['path'])==record['sha256'];_,dd=live_sorter.predict(model,cv2.imread(record['path']),config,'cpu')
   predictions[record['time']]=[{'box':list(d.box),'confidence':d.confidence,'class_id':d.class_id,'class_name':NAMES[d.class_id],'passes_60_percent':d.confidence>=.60} for d in dd]
  result['models'][name]={'checkpoint':proof,'predictions':predictions}
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
