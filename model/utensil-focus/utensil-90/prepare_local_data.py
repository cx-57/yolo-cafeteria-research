from pathlib import Path
import json,hashlib,yaml
root=Path.cwd();src=root/'data/utensil-only/coverage-v12';dest=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/utensil-v12');m=json.loads((src/'manifest.json').read_text())
for i,r in enumerate(m['records']):
 image=src/'images'/r['split']/r['name'];out=dest/'images'/r['split']/r['name'];out.parent.mkdir(parents=True,exist_ok=True)
 if not out.exists():
  raw=image.read_bytes();assert hashlib.sha256(raw).hexdigest()==r['sha'];out.write_bytes(raw)
 else:assert hashlib.sha256(out.read_bytes()).hexdigest()==r['sha']
 label=dest/'labels'/r['split']/Path(r['name']).with_suffix('.txt');label.parent.mkdir(parents=True,exist_ok=True);label.write_text('\n'.join(r['rows'])+ ('\n' if r['rows'] else ''))
 if i%100==0:print('Localized',i,'of',len(m['records']),flush=True)
(dest/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');(dest/'data.yaml').write_text(yaml.safe_dump(dict(path=str(dest),train='images/train',val='images/val',names=['utensil','condiment_packet']),sort_keys=False));(dest/'utensil.yaml').write_text(yaml.safe_dump(dict(path=str(dest),train='images/train',val='images/val',names=['utensil']),sort_keys=False));print('LOCAL_DATASET_READY',dest,flush=True)
