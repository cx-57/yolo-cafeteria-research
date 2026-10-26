"""Stage previously uninspected official-source photos; no admission or inference."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import random
from PIL import Image,ImageOps,ImageDraw
RUN=Path(__file__).resolve().parent;ROOT=RUN.parent;SOURCE=ROOT/'openimages-expansion-staging-v37'
spec=importlib.util.spec_from_file_location('v47_fetch',ROOT/'openimages-staging-v32/fetch_candidates.py');fetcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(fetcher)
fetcher.CACHE=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-openimages-drink-staging-v47')
def main():
 excluded=set()
 paths=[SOURCE/'novel-complete-boxes.json',SOURCE/'novel-photo-metadata.json']
 for folder in ['openimages-staging-v32','openimages-expansion-staging-v37']:
  p=ROOT/folder/'downloads.json';paths.append(p);excluded.update(r['id'] for r in json.loads(p.read_text()))
 p=ROOT/'monitor-campaign-v34/monitor-source-proposal.json';paths.append(p);excluded.update(r['id'] for r in json.loads(p.read_text())['records'])
 groups=json.loads(paths[0].read_text())['groups'];metadata={r['ImageID']:r for r in json.loads(paths[1].read_text())['found']}
 choices=[]
 for g in groups:
  m=metadata[g['id']]
  if g['id'] in excluded or m['Rotation'] not in ('0','0.0') or not m['License'].startswith('https://creativecommons.org/licenses/'):continue
  if not 1<=len(g['annotations'])<=6 or any(a['IsGroupOf']!='0' or a['IsDepiction']!='0' for a in g['annotations']):continue
  choices.append(dict(g,photo_metadata=m))
 random.Random(1143).shuffle(choices);selected=choices[:80]
 assert len(selected)==80 and not ({r['id'] for r in selected}&excluded)
 declaration=dict(seed=1143,candidate_limit=80,source_input_shas={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},fetcher_sha=hashlib.sha256(Path(fetcher.__file__).read_bytes()).hexdigest(),stage_code_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),selected_ids=[r['id'] for r in selected],excluded_previously_downloaded=True,no_model_outputs_used=True,source_documentation='https://storage.googleapis.com/openimages/web/download_v7.html',training_admission=False)
 q=RUN/'source-selection.json'
 if q.exists():assert json.loads(q.read_text())==declaration
 else:q.write_text(json.dumps(declaration,indent=2)+'\n')
 rows=[]
 with ThreadPoolExecutor(max_workers=4) as pool:
  for row in pool.map(fetcher.fetch,selected):rows.append(row);print('DRINK_SOURCE',len(rows),row['id'],row['status'],flush=True)
 (RUN/'downloads.json').write_text(json.dumps(rows,indent=2)+'\n')
 for start in range(0,len(rows),6):
  sheet=Image.new('RGB',(1536,1100),'white');d=ImageDraw.Draw(sheet)
  for j,r in enumerate(rows[start:start+6]):
   x=j%3*512;y=j//3*550
   if r['status']=='unreviewed_generic_bottle':
    im=Image.open(r['image']).convert('RGB');draw=ImageDraw.Draw(im)
    for k,a in enumerate(r['annotations']):
     xx,z,yy,t=[float(a[key]) for key in ['XMin','XMax','YMin','YMax']];draw.rectangle((xx*im.width,yy*im.height,z*im.width,t*im.height),outline='red',width=2);draw.text((xx*im.width,yy*im.height),str(k),fill='red',stroke_width=1,stroke_fill='white')
    im=ImageOps.contain(im,(504,495));sheet.paste(im,(x,y))
   d.text((x+3,y+502),f"{start+j}: {r['id']} {len(r['annotations'])} boxes",fill='black')
  sheet.save(RUN/f'preview-{start:03}.jpg',quality=93)
 print('SOURCE_STAGED_NO_ADMISSION',len(rows),flush=True)
if __name__=='__main__':main()
