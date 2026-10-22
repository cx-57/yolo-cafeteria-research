"""Stage unused official TACO clear-plastic-bottle photos for visual review."""
from pathlib import Path
from collections import Counter
import hashlib,json,shutil,subprocess
from PIL import Image,ImageDraw,ImageOps
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[2]
DATA=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-capture-validation')
CACHE=DATA.parent/'bottle-taco-new-20261022';CACHE.mkdir(exist_ok=True)
old=ROOT/'data/focus-three-class-improvement/bottle-candidates'
manifest=json.loads((DATA/'manifest.json').read_text())
known_sha={r['sha'] for r in manifest['records']}
known_names='\n'.join(r['name'] for r in manifest['records'])
records=[]
for r in json.loads((old/'candidates.json').read_text()):
    if r['source']!='TACO' or r['status']!='pending_full_three_class_review':continue
    if r['sha256'] in known_sha or r['id'] in known_names:continue
    target=CACHE/(r['id']+'.jpg')
    if not target.exists() and target.with_suffix('.png').exists():target=target.with_suffix('.png')
    if not target.exists():
        source=ROOT/r['path']
        if source.exists() and not getattr(source.stat(),'st_flags',0)&0x40000000:shutil.copy2(source,target)
        else:subprocess.run(['curl','-fsSL','--retry','2','--max-time','40',r['url'],'-o',str(target)],check=True)
    digest=hashlib.sha256(target.read_bytes()).hexdigest();assert digest==r['sha256']
    with Image.open(target) as im:
        im.load();dimensions=im.size
        orientation=im.getexif().get(274,1)
    if orientation!=1 or dimensions!=(r['original']['width'],r['original']['height']):
        records.append(dict(id=r['id'],image=str(target),sha=digest,status='orientation_or_dimensions_review_hold',orientation=orientation,dimensions=dimensions,original=r['original']))
        continue
    with Image.open(target) as im:
        actual_format=im.format
    if actual_format=='PNG':
        renamed=target.with_suffix('.png')
        target.replace(renamed);target=renamed
    records.append(dict(id=r['id'],image=str(target),sha=digest,original=r['original'],annotations=[a for a in r['annotations'] if a['category_id']==5],all_source_annotations=r['annotations'],source=r['source_url'],url=r['url'],license_status=r['license_status'],status='unreviewed',source_annotation_sha=hashlib.sha256((old/'taco_annotations.json').read_bytes()).hexdigest()))
(RUN/'taco-downloads.json').write_text(json.dumps(records,indent=2)+'\n')
good=[r for r in records if r['status']=='unreviewed']
print('TACO_STAGED',len(good),'BOXES',sum(len(r['annotations']) for r in good),'HOLDS',len(records)-len(good),flush=True)
records=good
for start in range(0,len(records),6):
    sheet=Image.new('RGB',(1536,1200),'white');d=ImageDraw.Draw(sheet)
    for j,r in enumerate(records[start:start+6]):
        im=Image.open(r['image']).convert('RGB');draw=ImageDraw.Draw(im)
        for k,a in enumerate(r['annotations']):
            x,y,w,h=a['bbox'];draw.rectangle((x,y,x+w,y+h),outline='red',width=max(3,im.width//300));draw.text((x,y),str(k),fill='red',stroke_width=2,stroke_fill='white')
        im=ImageOps.contain(im,(504,545));x=j%3*512;y=j//3*600;sheet.paste(im,(x,y));d.text((x+2,y+550),f"{start+j}: {r['id']} {len(r['annotations'])} boxes",fill='black')
    sheet.save(RUN/f'preview-{start:03d}.jpg',quality=90)
