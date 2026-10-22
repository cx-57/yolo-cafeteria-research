from pathlib import Path
import json,hashlib,collections,yaml
from PIL import Image
from v2 import dataset_fingerprint
from prepare import IMAGE_SUFFIXES
root=Path.cwd();source=root/'data/training-v2-codex-reviewed';build=json.loads((source/'v2_build.json').read_text());expected=build['dataset_sha256'];actual=dataset_fingerprint(source/'data.yaml');assert actual==expected,('Source dataset changed',actual,expected)
data=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/food-carton-container-specialist');data.mkdir(parents=True,exist_ok=False);mapping={0:0,4:1,6:2};names=['food','hinged_container','drink_carton'];records=[];counts={};hash_splits={};pixel_splits={};byte_overlap=[];pixel_overlap=[]
for split in ['train','val']:
 counter=collections.Counter();images=sorted(p for p in (source/'images'/split).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
 for i,image in enumerate(images):
  raw=image.read_bytes();sha=hashlib.sha256(raw).hexdigest();label=source/'labels'/split/(image.stem+'.txt');original_rows=label.read_text().splitlines();rows=[str(mapping[int(row.split()[0])])+' '+row.split(' ',1)[1] for row in original_rows if int(row.split()[0]) in mapping]
  dest=data/'images'/split/image.name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw);target=data/'labels'/split/label.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text('\n'.join(rows)+ ('\n' if rows else ''))
  with Image.open(dest) as im:
   rgb=im.convert('RGB');pixel_sha=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
  if sha in hash_splits and hash_splits[sha]!=split:byte_overlap.append(image.name)
  if pixel_sha in pixel_splits and pixel_splits[pixel_sha]!=split:pixel_overlap.append(image.name)
  hash_splits[sha]=split;pixel_splits[pixel_sha]=split
  records.append(dict(name=image.name,split=split,sha=sha,pixels=pixel_sha,rows=rows,source_label_sha=hashlib.sha256(label.read_bytes()).hexdigest()))
  counter.update(int(row.split()[0]) for row in rows)
  if i%500==0:print('Prepared',split,i,'of',len(images),flush=True)
 counts[split]=dict(images=len(images),objects={n:counter[j] for j,n in enumerate(names)})
 assert len(images)==build['final_audit']['image_counts'][split]
 for old_id,new_id in mapping.items():assert counter[new_id]==build['final_audit']['instances'][split][names[new_id]]
assert not byte_overlap and not pixel_overlap,('Cross-split image copies detected',byte_overlap,pixel_overlap)
(data/'data.yaml').write_text(yaml.safe_dump(dict(path=str(data),train='images/train',val='images/val',names=names),sort_keys=False))
manifest=dict(source=str(source),source_fingerprint=actual,records=records,counts=counts,class_mapping=mapping,names=names,train_val_byte_overlap=byte_overlap,train_val_decoded_pixel_overlap=pixel_overlap,source_annotation_status='Provisional inherited reviewed annotations; fresh inference does not independently certify semantic completeness.',independent_test=False,purpose='Fresh evaluation projection; original images, source labels and split assignments preserved.')
(data/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('DATA_READY',json.dumps(counts),flush=True)
