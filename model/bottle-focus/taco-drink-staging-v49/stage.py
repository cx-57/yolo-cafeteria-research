"""Stage a fresh fixed source-only litter sample; no admission or model inference."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import hashlib
import json
import random
from PIL import Image,ImageOps,ImageDraw
RUN=Path(__file__).resolve().parent;ROOT=RUN.parent;PREV=ROOT/'taco-expansion-staging-v38'
spec=importlib.util.spec_from_file_location('v49_litter_fetch',PREV/'stage.py');source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
source.CACHE=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-taco-drink-staging-v49')
def main():
 source.CACHE.mkdir(exist_ok=True);snapshot=PREV/'annotations-unofficial.json';raw=snapshot.read_bytes();data=json.loads(raw)
 original=ROOT.parents[1]/'data/focus-three-class-improvement/bottle-candidates/taco_annotations.json'
 old=json.loads(original.read_text());old_downloads=PREV/'downloads.json';previous=json.loads(old_downloads.read_text())
 excluded={source.photo_id(r['flickr_url']) for r in old['images']}|{source.photo_id(r['original']['flickr_url']) for r in previous}
 annotations={}
 for a in data['annotations']:annotations.setdefault(a['image_id'],[]).append(a)
 pool=[]
 for im in data['images']:
  boxes=annotations.get(im['id'],[]);clear=[a for a in boxes if a['category_id']==5]
  if 'img.tacodataset.org' not in im['flickr_url'] or source.photo_id(im['flickr_url']) in excluded:continue
  if not 1<=len(clear)<=6 or any(a['category_id']==4 for a in boxes):continue
  pool.append(dict(id=f"taco_unofficial_{im['id']}",original=im,annotations=clear,all_source_annotations=boxes,source_annotation_sha=hashlib.sha256(raw).hexdigest(),source_url='https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations_unofficial.json',source_quality='Community annotations not reviewed by TACO maintainers; independent scope/box/exhaustiveness QA required'))
 random.Random(1143).shuffle(pool);selected=pool[:60];assert len(selected)==60
 declaration=dict(seed=1143,limit=60,selected_ids=[r['id'] for r in selected],source_snapshot_sha=hashlib.sha256(raw).hexdigest(),previous_downloads_sha=hashlib.sha256(old_downloads.read_bytes()).hexdigest(),old_official_snapshot_sha=hashlib.sha256(original.read_bytes()).hexdigest(),fetcher_sha=hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest(),stage_code_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),no_model_outputs_used=True,training_admission=False)
 path=RUN/'source-selection.json'
 if path.exists():assert json.loads(path.read_text())==declaration
 else:path.write_text(json.dumps(declaration,indent=2)+'\n')
 rows=[]
 with ThreadPoolExecutor(max_workers=3) as pool:
  for row in pool.map(source.fetch,selected):rows.append(row);print('NEW_LITTER_SOURCE',len(rows),row['id'],row['status'],flush=True)
 (RUN/'downloads.json').write_text(json.dumps(rows,indent=2)+'\n')
 for start in range(0,len(rows),6):
  sheet=Image.new('RGB',(1536,1100),'white');d=ImageDraw.Draw(sheet)
  for j,row in enumerate(rows[start:start+6]):
   x=j%3*512;y=j//3*550
   if row['status']=='unreviewed_staged_only':
    im=Image.open(row['image']).convert('RGB');draw=ImageDraw.Draw(im)
    for a in row['annotations']:
     xx,yy,w,h=a['bbox'];draw.rectangle((xx,yy,xx+w,yy+h),outline='red',width=max(2,im.width//400))
    im=ImageOps.contain(im,(504,495));sheet.paste(im,(x,y))
   d.text((x+3,y+502),f"{start+j}: {row['id']} {row['status']}",fill='black')
  sheet.save(RUN/f'preview-{start:03}.jpg',quality=93)
 print('FRESH_LITTER_STAGED',sum(r['status']=='unreviewed_staged_only' for r in rows),len(rows),flush=True)
if __name__=='__main__':main()
