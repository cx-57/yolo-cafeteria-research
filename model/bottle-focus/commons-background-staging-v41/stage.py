"""Stage attributed Commons litter photos; categories never prove bottle absence."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import random
import subprocess
import threading
from PIL import Image,ImageOps,ImageDraw

RUN=Path(__file__).resolve().parent
CACHE=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-commons-background-v41')
AGENT='CafeteriaBottleResearch/1.0 (+https://github.com/cx-57/yolo-cafeteria-research)'
limited=threading.Event()

def fetch(row):
 result=dict(row,training_admission=False)
 if limited.is_set():return dict(result,status='network_rate_limit_hold')
 try:
  info=row['commons_record']['imageinfo'][0];url=info.get('thumburl',info['url']);path=CACHE/f"{row['id']}.image"
  if not path.exists():
   partial=path.with_suffix('.partial');response=subprocess.run(['curl','-fsSL','--max-time','30','-A',AGENT,'-w','%{http_code}',url,'-o',str(partial)],capture_output=True)
   if response.stdout.decode().strip()=='429' or b'429' in response.stderr:limited.set()
   response.check_returncode();partial.replace(path)
  with Image.open(path) as opened:
   opened.load();assert opened.getexif().get(274,1)==1
   expected=(info.get('thumbwidth',info['width']),info.get('thumbheight',info['height']));assert opened.size==expected
   suffix={'JPEG':'.jpg','PNG':'.png'}[opened.format];picture=opened.convert('RGB');pixel=hashlib.sha256(str(picture.size).encode()+picture.tobytes()).hexdigest()
  destination=path.with_suffix(suffix);path.replace(destination)
  return dict(result,image=str(destination),sha=hashlib.sha256(destination.read_bytes()).hexdigest(),pixels=pixel,width=expected[0],height=expected[1],source_image_url=url,status='unreviewed_staged_only')
 except Exception as exc:return dict(result,status='download_or_geometry_hold',error=str(exc))

if __name__=='__main__':
 CACHE.mkdir(exist_ok=True);pages={}
 for name in ['cigarette-source.json','litter-source.json']:
  raw=(RUN/name).read_bytes();data=json.loads(raw)
  for row in data.get('query',{}).get('pages',{}).values():
   if not row.get('imageinfo'):continue
   info=row['imageinfo'][0];meta=info.get('extmetadata',{});license_url=meta.get('LicenseUrl',{}).get('value','');artist=meta.get('Artist',{}).get('value','')
   if not license_url.startswith(('https://creativecommons.org/','http://creativecommons.org/')) or not artist:continue
   if not row['title'].lower().endswith(('.jpg','.jpeg','.png')):continue
   pages[row['pageid']]=dict(id='commons_'+str(row['pageid']),commons_record=row,source_api_snapshot=name,source_api_sha=hashlib.sha256(raw).hexdigest(),license_url=license_url,artist_html=artist,source_quality='Commons category membership is only a candidate filter; two independent whole-photo absence reviews and overlap screening are required')
 selected=sorted(pages.values(),key=lambda r:r['id']);random.Random(20261025).shuffle(selected);selected=selected[:80];rows=[]
 with ThreadPoolExecutor(max_workers=2) as pool:
  for row in pool.map(fetch,selected):rows.append(row);print('COMMONS_SOURCE',len(rows),row['id'],row['status'],flush=True)
 (RUN/'downloads.json').write_text(json.dumps(rows,indent=2)+'\n');good=[(i,r) for i,r in enumerate(rows) if r['status']=='unreviewed_staged_only']
 for start in range(0,len(good),6):
  sheet=Image.new('RGB',(1536,1100),'white');d=ImageDraw.Draw(sheet)
  for j,(i,r) in enumerate(good[start:start+6]):
   im=ImageOps.contain(Image.open(r['image']).convert('RGB'),(504,495));x=j%3*512;y=j//3*550;sheet.paste(im,(x,y));d.text((x+3,y+502),f"{i}: {r['id']}",fill='black')
  sheet.save(RUN/f'preview-{start:03}.jpg',quality=93)
 print('COMMONS_STAGED_ONLY',len(good),'OF',len(rows),flush=True)
