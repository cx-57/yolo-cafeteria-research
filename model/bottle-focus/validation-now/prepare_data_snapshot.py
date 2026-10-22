from pathlib import Path
import json,hashlib,copy,yaml
import focus_three_class as f
import utensil_packet as s
root=Path.cwd();src=root/'data/focus-three-class-improvement/capture-grouped-clean';m=f.verify(src)
data=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-capture-validation');data.mkdir(parents=True,exist_ok=True);records=copy.deepcopy(m['records'])
for r in records:
 original=src/'images'/r['split']/r['name'];out=data/'images'/r['split']/r['name'];out.parent.mkdir(parents=True,exist_ok=True)
 if not out.exists():
  raw=original.read_bytes();assert hashlib.sha256(raw).hexdigest()==r['sha'];out.write_bytes(raw)
 r['rows']=['0 '+row.split(' ',1)[1] for row in r['rows'] if row.startswith('2 ')];label=data/'labels'/r['split']/Path(r['name']).with_suffix('.txt');label.parent.mkdir(parents=True,exist_ok=True);label.write_text('\n'.join(r['rows'])+ ('\n' if r['rows'] else ''))
s.NAMES=['bottle'];s.validate(records,data)
(data/'data.yaml').write_text(yaml.safe_dump(dict(path=str(data),train='images/train',val='images/val',names=['bottle']),sort_keys=False));s.validate_configuration(data,records)
manifest=dict(records=records,source=str(src),source_fingerprint=m['fingerprint'],validation_protocol=m.get('validation_protocol'),validation_images=sum(r['split']=='val' for r in records),validation_bottle_objects=sum(len(r['rows']) for r in records if r['split']=='val'),validation_images_and_bottle_boxes_preserved=True)
s.save(data/'manifest.json',manifest);print('BOTTLE_VALIDATION_READY',manifest['validation_images'],manifest['validation_bottle_objects'],flush=True)
