from pathlib import Path
import json,collections
from PIL import Image,ImageDraw,ImageOps
root=Path.cwd();run=root/'model/bottle-focus/gentle-feature-adaptation-v6';m=json.loads(Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-capture-validation/manifest.json').read_text());val=[r for r in m['records'] if r['split']=='val'];preds=json.loads((run/'evaluations/selected-tta640/predictions.json').read_text());by=collections.defaultdict(list)
for p in preds:by[str(p['image_id'])].append(p)
def iou(a,b):
 inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-12)
fp=[];miss=[];potential=0;total=0
for r in val:
 path=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-capture-validation/images/val')/r['name'];im=Image.open(path);w,h=im.size;gt=[]
 for row in r['rows']:
  c,x,y,bw,bh=map(float,row.split());gt.append(((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h))
 pp=by.get(path.stem,[]);bb=[(p['bbox'][0],p['bbox'][1],p['bbox'][0]+p['bbox'][2],p['bbox'][1]+p['bbox'][3]) for p in pp]
 for box in gt:
  total+=1;best=max((iou(box,b) for b in bb),default=0);potential+=best>=.5
  if best<.5:miss.append(dict(name=r['name'],box=box,best_iou=best))
 for p,b in zip(pp,bb):
  overlap=max((iou(b,g) for g in gt),default=0)
  if overlap<.5 and p['score']>=.25:fp.append(dict(name=r['name'],box=b,score=p['score'],max_iou=overlap))
fp.sort(key=lambda x:x['score'],reverse=True)
print('Potential recall at IoU .5 across saved proposals:',potential,'/',total);print('False positives at confidence .25:',len(fp));print('Top FP sources:',collections.Counter(x['name'].split('__')[0] for x in fp));out=run/'phase8-error-audit';out.mkdir(exist_ok=True)
(out/'audit.json').write_text(json.dumps(dict(potential_matches=potential,total=total,false_positives=fp,missed=miss),indent=2)+'\n')
for kind,items in [('false-positives',fp[:18]),('misses',miss)]:
 for start in range(0,len(items),9):
  sheet=Image.new('RGB',(1536,1212),'white');d=ImageDraw.Draw(sheet)
  for j,item in enumerate(items[start:start+9]):
   p=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-capture-validation/images/val')/item['name'];im=Image.open(p).convert('RGB');ImageDraw.Draw(im).rectangle(tuple(item['box']),outline='red',width=max(2,im.width//180));thumb=ImageOps.contain(im,(504,370));x=j%3*512;y=j//3*404;sheet.paste(thumb,(x,y));d.text((x+2,y+374),f"{start+j}: {item['name'][-40:]} {item.get('score',0):.2f}",fill='black')
  sheet.save(out/f'{kind}-{start:02d}.jpg')
