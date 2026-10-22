"""Stage new LVIS plastic-drink-bottle candidates, without admitting them."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import random
import os
import re
import subprocess
from PIL import Image,ImageOps,ImageDraw

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[2]
CACHE=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-lvis-new-20261022')
OLD=ROOT/'data/focus-three-class-improvement/bottle-candidates'
INDEX=json.loads((OLD/'additional_lvis_training_index.json').read_text())
MANIFEST=json.loads(Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-capture-validation/manifest.json').read_text())

def fetch(r):
    path=CACHE/(r['id']+'.jpg');path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        urls=[r['original']['coco_url'].replace('http:','https:'),r['original']['flickr_url'].replace('http:','https:')]
        for url in urls:
            result=subprocess.run(['curl','-fsSL','--max-time','20',url,'-o',str(path)],capture_output=True)
            if result.returncode==0:break
        else:return dict(id=r['id'],status='download_failed')
    try:
        with Image.open(path) as im:
            im.load();display=ImageOps.exif_transpose(im)
            assert display.size==(r['original']['width'],r['original']['height'])
    except Exception as e:return dict(id=r['id'],status='invalid_image',error=str(e))
    return dict(r,image=str(path),sha=hashlib.sha256(path.read_bytes()).hexdigest(),status='unreviewed',source='https://www.lvisdataset.org',archive_sha256=INDEX['archive_sha256'])

if __name__=='__main__':
    known=set()
    for r in MANIFEST['records']:
        known.update(int(x) for x in re.findall(r'lvis_(\d+)',r['name']))
    for p in (OLD/'images').glob('*.jpg'):
        known.update(int(x) for x in re.findall(r'lvis_(\d+)',p.name))
    candidates=[r for r in INDEX['candidates'] if r['original']['id'] not in known and 1162 not in r['original'].get('not_exhaustive_category_ids',[]) and 1<=len(r['annotations'])<=8]
    random.Random(42).shuffle(candidates)
    offset=int(os.environ.get('BOTTLE_OFFSET','0'))
    count=int(os.environ.get('BOTTLE_COUNT','60'))
    prefix=os.environ.get('BOTTLE_ROUND','lvis')
    selected=candidates[offset:offset+count]
    records=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for r in pool.map(fetch,selected):
            records.append(r);print('LVIS_DOWNLOAD',len(records),r['id'],r['status'],flush=True)
    (RUN/(prefix+'-downloads.json')).write_text(json.dumps(records,indent=2)+'\n')
    good=[r for r in records if r['status']=='unreviewed']
    for start in range(0,len(good),6):
        sheet=Image.new('RGB',(1536,1000),'white');d=ImageDraw.Draw(sheet)
        for j,r in enumerate(good[start:start+6]):
            im=ImageOps.exif_transpose(Image.open(r['image'])).convert('RGB');draw=ImageDraw.Draw(im)
            for k,a in enumerate(r['annotations']):
                x,y,w,h=a['bbox'];draw.rectangle((x,y,x+w,y+h),outline='red',width=2);draw.text((x,y),str(k),fill='red',stroke_width=1,stroke_fill='white')
            im=ImageOps.contain(im,(504,450));x=j%3*512;y=j//3*500;sheet.paste(im,(x,y));d.text((x+2,y+455),f"{start+j}: {r['id']} {len(r['annotations'])} boxes",fill='black')
        sheet.save(RUN/f'{prefix}-preview-{start:03d}.jpg',quality=90)
