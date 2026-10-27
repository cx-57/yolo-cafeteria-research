"""Single-class sauce and dry condiment packet specialist with isolated reviewed data."""
import hashlib
import json
import math
from pathlib import Path
import shutil

from storage import ROOT, DATA, RUNTIME as RUNTIME_ROOT, resolve_image
RUNTIME=RUNTIME_ROOT/'packet-broad-v16'
REVIEWS=DATA/'packet-focus/broad-v16/reviews.json'
REPORTS=ROOT/'runs/data'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def yolo_row(box):
 x1,y1,x2,y2=box
 return f'0 {(x1+x2)/2:.9f} {(y1+y2)/2:.9f} {x2-x1:.9f} {y2-y1:.9f}'
def validate_records(records):
 seen={field:{} for field in ('group','sha256','pixels')};sources={};names=set()
 for r in records:
  if r['split'] not in ('train','val') or r['name'] in names:raise ValueError('Invalid split or duplicate name')
  names.add(r['name'])
  prov=r.get('provenance',{})
  for field in ('source_sha1','pageid'):
   key=prov.get(field)
   if key:
    key=(field,str(key))
    if key in sources and sources[key]!=r['split']:raise ValueError('Cross-split canonical source leakage')
    sources[key]=r['split']
  for field,values in seen.items():
   key=r[field]
   if not key:raise ValueError('Missing identity '+field)
   if key in values and values[key]!=r['split']:raise ValueError('Cross-split leakage: '+field)
   values[key]=r['split']
  for box in r['boxes']:
   if len(box)!=4 or not all(math.isfinite(v) and 0<=v<=1 for v in box) or not (box[0]<box[2] and box[1]<box[3]):raise ValueError('Invalid box')

def verify():
 from PIL import Image
 import yaml
 d=json.loads((RUNTIME/'manifest.json').read_text());validate_records(d['records'])
 if digest(REVIEWS)!=d['review_sha256'] or json.loads(REVIEWS.read_text())!=d['records']:raise ValueError('Review ledger changed')
 expected=dict(path=str(RUNTIME),train='images/train',val='images/val',names=['condiment_packet'])
 if yaml.safe_load((RUNTIME/'data.yaml').read_text())!=expected:raise ValueError('Dataset YAML changed')
 for split in ('train','val'):
  for kind in ('images','labels'):
   expected_names={r['name'] if kind=='images' else str(Path(r['name']).with_suffix('.txt')) for r in d['records'] if r['split']==split}
   directory=RUNTIME/kind/split
   actual={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
   if actual!=expected_names:raise ValueError('Dataset membership changed')
 for r in d['records']:
  p=RUNTIME/'images'/r['split']/r['name']
  if digest(p)!=r['sha256']:raise ValueError('Image changed '+r['name'])
  im=Image.open(p).convert('RGB')
  if hashlib.sha256(im.tobytes()).hexdigest()!=r['pixels']:raise ValueError('Pixels changed')
  label=RUNTIME/'labels'/r['split']/Path(r['name']).with_suffix('.txt')
  if label.read_text()!=''.join(yolo_row(b)+'\n' for b in r['boxes']):raise ValueError('Label changed')
 return d

def build():
 import yaml
 from PIL import Image
 if (RUNTIME/'manifest.json').exists():raise ValueError('Dataset exists; verify rather than overwrite')
 records=json.loads(REVIEWS.read_text());validate_records(records)
 for r in records:
  source=resolve_image(r['image_path']);im=Image.open(source).convert('RGB')
  if digest(source)!=r['sha256'] or hashlib.sha256(im.tobytes()).hexdigest()!=r['pixels']:raise ValueError('Review identity changed')
  for kind in ('images','labels'):(RUNTIME/kind/r['split']).mkdir(parents=True,exist_ok=True)
  shutil.copyfile(source,RUNTIME/'images'/r['split']/r['name'])
  (RUNTIME/'labels'/r['split']/Path(r['name']).with_suffix('.txt')).write_text(''.join(yolo_row(b)+'\n' for b in r['boxes']))
 (RUNTIME/'data.yaml').write_text(yaml.safe_dump(dict(path=str(RUNTIME),train='images/train',val='images/val',names=['condiment_packet'])))
 manifest=dict(records=records,review_sha256=digest(REVIEWS),ontology='Physical flexible single-serving sauce, salt, pepper and sugar packets; exclude rigid tubs, bottles, bulk packaging and tea bags',independent_test=False,annotation='AI-assisted full-image review; not independently human certified')
 save(RUNTIME/'manifest.json',manifest);save(REPORTS/'dataset.json',dict(review_sha256=manifest['review_sha256'],counts={s:dict(images=sum(r['split']==s for r in records),boxes=sum(len(r['boxes']) for r in records if r['split']==s)) for s in ('train','val')},independent_test=False))
 verify()
