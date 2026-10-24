"""Download a bounded public-source inspection sample; no dataset admission or training."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import subprocess
from PIL import Image,ImageOps,ImageDraw

RUN=Path(__file__).resolve().parent
CACHE=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-openimages-staging-v32')

def fetch(record):
    path=CACHE/f"openimages__train__{record['id']}.jpg";path.parent.mkdir(parents=True,exist_ok=True)
    url=f"https://open-images-dataset.s3.amazonaws.com/train/{record['id']}.jpg"
    result=dict(record,canonical_image_url=url,training_admission=False)
    try:
        if not path.exists():
            tmp=path.with_suffix('.jpg.partial');subprocess.run(['curl','-fsSL','--max-time','25',url,'-o',str(tmp)],check=True,capture_output=True);tmp.replace(path)
        raw=path.read_bytes();assert raw[-2:]==b'\xff\xd9'
        with Image.open(path) as opened:
            assert opened.format=='JPEG' and opened.getexif().get(274,1)==1
            picture=ImageOps.exif_transpose(opened).convert('RGB');picture.load();w,h=picture.size
            pixel=hashlib.sha256(str(picture.size).encode()+picture.tobytes()).hexdigest()
        return dict(result,image=str(path),sha=hashlib.sha256(raw).hexdigest(),pixels=pixel,width=w,height=h,status='unreviewed_generic_bottle')
    except Exception as error:return dict(result,status='download_or_geometry_failure',error=str(error))

if __name__=='__main__':
    groups=json.loads((RUN/'targeted-annotation-prefix.json').read_text())['groups'];metadata={r['ImageID']:r for r in json.loads((RUN/'targeted-photo-metadata.json').read_text())['found']};selected=[]
    groups=sorted(groups,key=lambda g:(-len({a['LabelName'] for a in g['positive_source_labels']}),g['id']))
    for g in groups:
        meta=metadata[g['id']]
        if meta['Rotation'] not in ('0','0.0') or not meta['License'].startswith('https://creativecommons.org/'):continue
        if not 1<=len(g['annotations'])<=6 or any(a['IsGroupOf']!='0' or a['IsDepiction']!='0' for a in g['annotations']):continue
        selected.append(dict(g,photo_metadata=meta))
        if len(selected)>=60:break
    with ThreadPoolExecutor(max_workers=4) as pool:
        records=[]
        for r in pool.map(fetch,selected):records.append(r);print('OPENIMAGES_SAMPLE',len(records),r['id'],r['status'],flush=True)
    (RUN/'targeted-downloads.json').write_text(json.dumps(records,indent=2)+'\n');good=[r for r in records if r['status']=='unreviewed_generic_bottle']
    for start in range(0,len(good),6):
        sheet=Image.new('RGB',(1536,1100),'white');d=ImageDraw.Draw(sheet)
        for j,r in enumerate(good[start:start+6]):
            im=Image.open(r['image']).convert('RGB');draw=ImageDraw.Draw(im)
            for k,a in enumerate(r['annotations']):
                x,z,y,w=[float(a[key]) for key in ('XMin','XMax','YMin','YMax')];draw.rectangle((x*im.width,y*im.height,z*im.width,w*im.height),outline='red',width=2);draw.text((x*im.width,y*im.height),str(k),fill='red',stroke_width=1,stroke_fill='white')
            im=ImageOps.contain(im,(504,495));x=j%3*512;y=j//3*550;sheet.paste(im,(x,y));d.text((x+3,y+502),f"{start+j}: {r['id']} {len(r['annotations'])} generic bottle boxes",fill='black')
        sheet.save(RUN/f'targeted-preview-{start:03}.jpg',quality=92)
    print('STAGED_ONLY',len(good),'NO_TRAINING_ADMISSION',flush=True)
