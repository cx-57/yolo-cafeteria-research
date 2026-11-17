import json,sys,collections
from pathlib import Path
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parents[2]/'scripts'))
from evaluate_sorting import iou
from specialists import NAMES
records=json.loads((R/'benchmark/freeze.json').read_text())['records'];out={}
for model in ['parent','control','candidate']:
 predictions=json.loads((R/f'evaluation-{model}/predictions.json').read_text());counts=collections.Counter();examples=[]
 for r in records:
  truth=[]
  for row in r['rows']:
   c,x,y,w,h=map(float,row.split());truth.append(([x-w/2,y-h/2,x+w/2,y+h/2],int(c)))
  unmatched=set(range(len(truth)))
  for d in sorted(predictions[r['name']],key=lambda x:-x[4]):
   if d[4]<.6:continue
   matches=[(iou(d[:4],truth[j][0]),j) for j in unmatched if truth[j][1]==d[5]];score,j=max(matches,default=(0,-1))
   if score>=.5:unmatched.remove(j);continue
   same=max([iou(d[:4],box) for box,c in truth if c==d[5]],default=0);anybest=max([(iou(d[:4],box),c) for box,c in truth],default=(0,None))
   kind='duplicate' if same>=.5 else 'wrong_class' if anybest[0]>=.5 else 'localization' if same>=.1 else 'background_or_missing_label'
   counts[(r['condition'],NAMES[int(d[5])],kind)]+=1
   examples.append({'image':r['image_path'],'name':r['name'],'condition':r['condition'],'prediction':d,'kind':kind,'same_class_iou':same,'best_any_class':anybest})
 out[model]={'counts':[{'condition':k[0],'class':k[1],'kind':k[2],'count':v} for k,v in counts.items()],'examples':examples}
(R/'precision-diagnosis.json').write_text(json.dumps(out,indent=2)+'\n')
for m,d in out.items():
 print(m,collections.Counter({kind:sum(x['count'] for x in d['counts'] if x['kind']==kind) for kind in ['duplicate','wrong_class','localization','background_or_missing_label']}))
 print('clean',[(x['class'],x['kind'],x['count']) for x in d['counts'] if x['condition']=='clean'])
