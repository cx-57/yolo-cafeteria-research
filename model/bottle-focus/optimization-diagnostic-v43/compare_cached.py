from pathlib import Path
from collections import defaultdict,Counter
import hashlib,json
from PIL import Image
RUN=Path('model/bottle-focus/optimization-diagnostic-v43');DATA=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-capture-validation');OLD=Path('model/bottle-focus/joint-metric-v30/evaluations/phase0-initial/predictions.json');NEW=Path('model/bottle-focus/reviewed-full-training-v42/evaluations/original-benchmark-95370c9b321b/predictions.json')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def iou(a,b):
 inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-12)
def evaluate(path):
 by=defaultdict(list)
 for p in json.loads(path.read_text()):by[p['file_name']].append(p)
 summaries=[]
 for record in json.loads((DATA/'manifest.json').read_text())['records']:
  if record['split']!='val':continue
  with Image.open(DATA/'images/val'/record['name']) as im:w,h=im.size
  gt=[]
  for row in record['rows']:
   _,x,y,bw,bh=map(float,row.split());gt.append(((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h))
  predictions=sorted([p for p in by[record['name']] if p['score']>=.25],key=lambda p:-p['score']);matched=set();fps=[];ious=[]
  for p in predictions:
   x,y,bw,bh=p['bbox'];box=(x,y,x+bw,y+bh);options=[(iou(box,g),i) for i,g in enumerate(gt) if i not in matched];best,index=max(options,default=(0,-1))
   if best>=.5:matched.add(index);ious.append(best)
   else:fps.append(dict(score=p['score'],box=box,best_unmatched_iou=best))
  summaries.append(dict(name=record['name'],source=record['name'].split('__')[0],boxes=len(gt),TP=len(matched),FP=len(fps),FN=len(gt)-len(matched),matched_ious=ious,false_positives=fps,matched_gt_indices=sorted(matched)))
 return summaries
old=evaluate(OLD);new=evaluate(NEW);assert [r['name'] for r in old]==[r['name'] for r in new]
summary={};sources=sorted({r['source'] for r in old})
for role,records in [('balanced',old),('v42',new)]:
 summary[role]={source:{k:sum(r[k] for r in records if r['source']==source) for k in ['boxes','TP','FP','FN']} for source in sources}
 summary[role]['all']={k:sum(r[k] for r in records) for k in ['boxes','TP','FP','FN']}
changes=[dict(name=a['name'],source=a['source'],before={k:a[k] for k in ['TP','FP','FN']},after={k:b[k] for k in ['TP','FP','FN']},lost_gt_indices=sorted(set(a['matched_gt_indices'])-set(b['matched_gt_indices']))) for a,b in zip(old,new) if (a['TP'],a['FP'])!=(b['TP'],b['FP'])]
report=dict(status='cached_predictions_only_no_new_benchmark_inference',confidence_for_error_inspection=.25,iou_for_error_inspection=.5,not_an_AP_recomputation=True,inputs={str(p):sha(p) for p in [OLD,NEW,DATA/'manifest.json']},summary=summary,changes=changes,records=dict(balanced=old,v42=new))
(RUN/'cached-error-comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2));print('Changed photos',len(changes),'lost matched boxes',sum(len(r['lost_gt_indices']) for r in changes))
