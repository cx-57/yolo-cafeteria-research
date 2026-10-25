"""Stage novel litter photos for independent review, never training admission."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlsplit
import hashlib
import json
import random
import re
import subprocess
from PIL import Image,ImageOps,ImageDraw

RUN=Path(__file__).resolve().parent
CACHE=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-taco-unofficial-v38')

def photo_id(url):return Path(urlsplit(url).path).name.split('_')[0].split('.')[0]

def fetch(row):
 result=dict(row,training_admission=False)
 try:
  identity=photo_id(row['original']['flickr_url']);assert identity.isdigit()
  page='https://www.flickr.com/photo.gne?id='+identity
  html=subprocess.run(['curl','-fsSL','--max-time','25',page],check=True,capture_output=True).stdout
  graphs=[json.loads(x) for x in re.findall(r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>',html.decode(),re.S)]
  objects=[r for g in graphs for r in g.get('@graph',[g]) if r.get('@type')=='ImageObject'];assert len(objects)==1
  meta=objects[0];assert photo_id(meta['contentUrl'])==identity
  result['photo_metadata']=dict(photo_id=identity,source_url=page,source_html_sha=hashlib.sha256(html).hexdigest(),photo_page=meta.get('acquireLicensePage'),license_url=meta.get('license'),author=meta.get('author'),retrieved_at=datetime.now(timezone.utc).isoformat())
  assert result['photo_metadata']['license_url'].startswith('https://creativecommons.org/licenses/')
  target=CACHE/f"{row['id']}.image"
  if not target.exists():
   partial=target.with_suffix('.partial');subprocess.run(['curl','-fsSL','--max-time','40',row['original']['flickr_url'],'-o',str(partial)],check=True,capture_output=True);partial.replace(target)
  with Image.open(target) as im:
   im.load();assert im.getexif().get(274,1)==1 and im.size==(row['original']['width'],row['original']['height'])
   suffix={'JPEG':'.jpg','PNG':'.png'}[im.format];picture=im.convert('RGB')
   pixels=hashlib.sha256(str(picture.size).encode()+picture.tobytes()).hexdigest()
  destination=target.with_suffix(suffix);target.replace(destination)
  return dict(result,image=str(destination),sha=hashlib.sha256(destination.read_bytes()).hexdigest(),pixels=pixels,status='unreviewed_staged_only')
 except Exception as exc:return dict(result,status='source_metadata_or_geometry_hold',error=str(exc))

if __name__=='__main__':
 CACHE.mkdir(exist_ok=True);source_path=RUN/'annotations-unofficial.json';raw=source_path.read_bytes();data=json.loads(raw)
 old_path=RUN.parents[2]/'data/focus-three-class-improvement/bottle-candidates/taco_annotations.json';old=json.loads(old_path.read_text())
 excluded={photo_id(r['flickr_url']) for r in old['images']};annotations={}
 for annotation in data['annotations']:annotations.setdefault(annotation['image_id'],[]).append(annotation)
 selected=[]
 for original in data['images']:
  boxes=annotations.get(original['id'],[]);clear=[r for r in boxes if r['category_id']==5]
  if 'img.tacodataset.org' not in original['flickr_url'] or photo_id(original['flickr_url']) in excluded:continue
  if not 1<=len(clear)<=6 or any(r['category_id']==4 for r in boxes):continue
  selected.append(dict(id=f"taco_unofficial_{original['id']}",original=original,annotations=clear,all_source_annotations=boxes,source_annotation_sha=hashlib.sha256(raw).hexdigest(),source_url='https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations_unofficial.json',source_quality='Community annotations not reviewed by TACO maintainers; independent scope/box/exhaustiveness QA required'))
 random.Random(1043).shuffle(selected);selected=selected[:120];rows=[]
 with ThreadPoolExecutor(max_workers=3) as pool:
  for row in pool.map(fetch,selected):rows.append(row);print('LITTER_SOURCE',len(rows),row['id'],row['status'],flush=True)
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
 print('LITTER_STAGING_COMPLETE_NO_TRAINING',sum(r['status']=='unreviewed_staged_only' for r in rows),flush=True)
