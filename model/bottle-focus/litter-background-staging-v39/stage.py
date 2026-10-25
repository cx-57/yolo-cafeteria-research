"""Stage fresh litter backgrounds; source class absence is never verified absence."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import random
from PIL import Image,ImageDraw,ImageOps

RUN=Path(__file__).resolve().parent;SOURCE=RUN.parent/'taco-expansion-staging-v38'
spec=importlib.util.spec_from_file_location('v39_fetch',SOURCE/'stage.py');fetcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(fetcher)
fetcher.CACHE=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-litter-background-v39')

if __name__=='__main__':
 fetcher.CACHE.mkdir(exist_ok=True);raw=(SOURCE/'annotations-unofficial.json').read_bytes();data=json.loads(raw)
 old=json.loads((RUN.parents[2]/'data/focus-three-class-improvement/bottle-candidates/taco_annotations.json').read_text());excluded={fetcher.photo_id(r['flickr_url']) for r in old['images']};annotations={}
 for row in data['annotations']:annotations.setdefault(row['image_id'],[]).append(row)
 candidates=[]
 for row in data['images']:
  boxes=annotations.get(row['id'],[])
  if not boxes or any(a['category_id'] in {4,5,6} for a in boxes):continue
  if 'img.tacodataset.org' not in row['flickr_url'] or fetcher.photo_id(row['flickr_url']) in excluded:continue
  candidates.append(dict(id=f"taco_unofficial_background_{row['id']}",original=row,annotations=[],all_source_annotations=boxes,source_annotation_sha=hashlib.sha256(raw).hexdigest(),source_url='https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations_unofficial.json',source_quality='Community source labels only; complete independent bottle-absence reviews required; no training admission'))
 random.Random(20261025).shuffle(candidates);candidates=candidates[:64];rows=[]
 with ThreadPoolExecutor(max_workers=3) as pool:
  for row in pool.map(fetcher.fetch,candidates):rows.append(row);print('LITTER_BACKGROUND',len(rows),row['id'],row['status'],flush=True)
 (RUN/'downloads.json').write_text(json.dumps(rows,indent=2)+'\n');good=[(i,r) for i,r in enumerate(rows) if r['status']=='unreviewed_staged_only']
 for start in range(0,len(good),6):
  sheet=Image.new('RGB',(1536,1100),'white');d=ImageDraw.Draw(sheet)
  for j,(i,r) in enumerate(good[start:start+6]):
   im=ImageOps.contain(Image.open(r['image']).convert('RGB'),(504,495));x=j%3*512;y=j//3*550;sheet.paste(im,(x,y));d.text((x+3,y+502),f"{i}: {r['id']}",fill='black')
  sheet.save(RUN/f'preview-{start:03}.jpg',quality=94)
 print('STAGED_BACKGROUNDS_UNREVIEWED',len(good),'NO_TRAINING_ADMISSION',flush=True)
