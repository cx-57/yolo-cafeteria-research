"""Stage a fixed source-only sample for visual review; no admission or training."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import importlib.util
import json
import random
from PIL import Image,ImageDraw,ImageOps

RUN=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v37_fetch',RUN.parent/'openimages-staging-v32/fetch_candidates.py')
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
source.CACHE=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-openimages-staging-v37')

if __name__=='__main__':
 groups=json.loads((RUN/'novel-complete-boxes.json').read_text())['groups']
 metadata={r['ImageID']:r for r in json.loads((RUN/'novel-photo-metadata.json').read_text())['found']}
 forbidden={r['id'] for r in json.loads((RUN.parent/'monitor-campaign-v34/monitor-source-proposal.json').read_text())['records']}
 selected=[]
 for group in groups:
  meta=metadata[group['id']]
  if group['id'] in forbidden or meta['Rotation'] not in ('0','0.0') or not meta['License'].startswith('https://creativecommons.org/licenses/'):continue
  if not 1<=len(group['annotations'])<=6 or any(a['IsGroupOf']!='0' or a['IsDepiction']!='0' for a in group['annotations']):continue
  selected.append(dict(group,photo_metadata=meta))
 random.Random(1043).shuffle(selected);selected=selected[:120]
 rows=[]
 with ThreadPoolExecutor(max_workers=4) as pool:
  for row in pool.map(source.fetch,selected):
   rows.append(row);print('SOURCE_SAMPLE',len(rows),row['id'],row['status'],flush=True)
 (RUN/'downloads.json').write_text(json.dumps(rows,indent=2)+'\n')
 # Preserve original indices even when a source download fails.
 for start in range(0,len(rows),6):
  sheet=Image.new('RGB',(1536,1100),'white');d=ImageDraw.Draw(sheet)
  for j,row in enumerate(rows[start:start+6]):
   x=j%3*512;y=j//3*550
   if row['status']=='unreviewed_generic_bottle':
    im=Image.open(row['image']).convert('RGB');boxes=ImageDraw.Draw(im)
    for k,a in enumerate(row['annotations']):
     xx,z,yy,t=[float(a[key]) for key in ['XMin','XMax','YMin','YMax']]
     boxes.rectangle((xx*im.width,yy*im.height,z*im.width,t*im.height),outline='red',width=2)
    im=ImageOps.contain(im,(504,495));sheet.paste(im,(x,y))
   d.text((x+3,y+502),f"{start+j}: {row['id']} {len(row['annotations'])} boxes",fill='black')
  sheet.save(RUN/f'preview-{start:03}.jpg',quality=93)
 print('STAGED_ONLY_NO_ADMISSION_NO_TRAINING',len(rows),flush=True)
