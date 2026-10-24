"""Stage remaining official source candidates only; never modify the active campaign."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from PIL import Image,ImageOps,ImageDraw

RUN=Path(__file__).resolve().parent;OI=RUN.parent/'openimages-staging-v32'
spec=importlib.util.spec_from_file_location('source_fetch',OI/'fetch_candidates.py');source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
if __name__=='__main__':
 groups=json.loads((OI/'targeted-annotation-prefix.json').read_text())['groups'];metadata={r['ImageID']:r for r in json.loads((OI/'targeted-photo-metadata.json').read_text())['found']};used={r['id'] for r in json.loads((OI/'targeted-downloads.json').read_text())};monitor={r['id'] for r in json.loads((RUN.parent/'monitor-campaign-v34/monitor-source-proposal.json').read_text())['records']};selected=[]
 for group in groups:
  meta=metadata[group['id']]
  if group['id'] in used|monitor or meta['Rotation'] not in ('0','0.0') or not meta['License'].startswith('https://creativecommons.org/'):continue
  if not 1<=len(group['annotations'])<=6 or any(a['IsGroupOf']!='0' or a['IsDepiction']!='0' for a in group['annotations']):continue
  selected.append(dict(group,photo_metadata=meta))
 with ThreadPoolExecutor(max_workers=4) as pool:
  rows=[]
  for row in pool.map(source.fetch,selected):rows.append(row);print('FOLLOWUP_SOURCE',len(rows),row['id'],row['status'],flush=True)
 (RUN/'additional-downloads.json').write_text(json.dumps(rows,indent=2)+'\n');good=[r for r in rows if r['status']=='unreviewed_generic_bottle']
 for start in range(0,len(good),6):
  sheet=Image.new('RGB',(1536,1100),'white');draw=ImageDraw.Draw(sheet)
  for j,row in enumerate(good[start:start+6]):
   im=Image.open(row['image']).convert('RGB');d=ImageDraw.Draw(im)
   for k,a in enumerate(row['annotations']):
    x,z,y,w=[float(a[key]) for key in ['XMin','XMax','YMin','YMax']];d.rectangle((x*im.width,y*im.height,z*im.width,w*im.height),outline='red',width=2);d.text((x*im.width,y*im.height),str(k),fill='red',stroke_width=1,stroke_fill='white')
   im=ImageOps.contain(im,(504,495));x=j%3*512;y=j//3*550;sheet.paste(im,(x,y));draw.text((x+3,y+502),f"{start+j}: {row['id']} {len(row['annotations'])} boxes",fill='black')
  sheet.save(RUN/f'additional-preview-{start:03}.jpg',quality=94)
 print('STAGED_ONLY',len(good),'ACTIVE_CAMPAIGN_UNCHANGED',flush=True)
