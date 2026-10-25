"""Prepare expanded training with both permanent monitors held out."""
from collections import Counter
from pathlib import Path
import ctypes
import hashlib
import json
import os
import yaml
import experiment
import sources
RUN=Path(__file__).resolve().parent;ROOT=RUN.parent
CACHE=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-reviewed-full-v42')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
v36=sources.module('v42_base_build',ROOT/'reviewed-neck-monitor-v36/build.py')
def save(path,obj):Path(path).write_text(json.dumps(obj,indent=2)+'\n')
def exposure(record):return 2 if record['source_type']=='openimages' else 8 if record['rows'] else 4
def clone(source,destination):
 destination.parent.mkdir(parents=True,exist_ok=True)
 if destination.exists():assert sha(destination)==sha(source);return
 lib=ctypes.CDLL('/usr/lib/libSystem.B.dylib',use_errno=True)
 assert lib.clonefile(os.fsencode(source),os.fsencode(destination),0)==0,'APFS clone failed'
 assert sha(destination)==sha(source)
def derive():
 base=v36.verify();compiled=sources.compile_all()
 old=Counter((ROOT/'reviewed-neck-monitor-v36/training-exposures.txt').read_text().splitlines())
 # Original training is verified recursively; new records were screened against every base image.
 forbidden_shas={r['image_sha'] for r in compiled['monitor_records']}
 forbidden_pixels={r['pixels'] for r in compiled['monitor_records']}
 for image in old:
  digest,pixels=sources.admission.hashes(image)
  assert digest not in forbidden_shas and pixels not in forbidden_pixels
 return base,compiled,old
def verify():
 base,compiled,counts=derive();audit=json.loads((RUN/'dataset.json').read_text())
 assert audit['source_input_shas']==compiled['bindings'] and audit['base_dataset_sha']==sha(ROOT/'reviewed-neck-monitor-v36/dataset.json')
 assert audit['monitor_records']==compiled['monitor_records']
 for source,record in zip(compiled['records'],audit['added'],strict=True):
  assert all(record[k]==v for k,v in source.items())
  assert sources.admission.hashes(record['image'])==(source['image_sha'],source['pixels'])
  assert experiment.label_path(Path(record['image'])).read_text().splitlines()==source['rows']
  counts[record['image']]+=exposure(source)
 listing=(RUN/'training-exposures.txt').read_text().splitlines()
 assert Counter(listing)==counts and sha(RUN/'training-exposures.txt')==audit['training_listing_sha']
 val=[]
 for record in compiled['monitor_records']:
  image=record.get('image',record['source_image']);label=experiment.label_path(Path(image))
  assert sources.admission.hashes(image)==(record['image_sha'],record['pixels'])
  assert label.read_text().splitlines()==record['rows'];val.append(image)
 assert (RUN/'monitor-images.txt').read_text().splitlines()==val
 assert audit['monitor_listing_sha']==sha(RUN/'monitor-images.txt')
 assert yaml.safe_load((RUN/'data.yaml').read_text())==audit['config']
 assert audit['training_photos']==len(counts) and audit['training_exposures']==len(listing)
 assert audit['training_bottle_boxes']==sum(len(experiment.label_path(Path(p)).read_text().splitlines()) for p in counts)
 assert audit['monitor_photos']==68 and audit['monitor_bottle_boxes']==35
 return audit
def main():
 assert not (RUN/'training-start.json').exists(),'Preparation locked after training starts'
 base,compiled,counts=derive();added=[]
 for source in compiled['records']:
  image=CACHE/'images/train'/Path(source['source_image']).name;label=experiment.label_path(image)
  clone(source['source_image'],image);label.parent.mkdir(parents=True,exist_ok=True)
  rows=experiment.validate_rows(source['rows'],image,['bottle']);assert rows==source['rows']
  label.write_text('\n'.join(rows)+('\n' if rows else ''))
  record=dict(source,image=str(image),label=str(label));added.append(record);counts[str(image)]+=exposure(source)
 listing=[p for p,n in counts.items() for _ in range(n)];(RUN/'training-exposures.txt').write_text('\n'.join(listing)+'\n')
 monitors=compiled['monitor_records'];val=[r.get('image',r['source_image']) for r in monitors];(RUN/'monitor-images.txt').write_text('\n'.join(val)+'\n')
 config=dict(path=str(CACHE),train=str(RUN/'training-exposures.txt'),val=str(RUN/'monitor-images.txt'),names=['bottle']);(RUN/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
 audit=dict(base_dataset_sha=sha(ROOT/'reviewed-neck-monitor-v36/dataset.json'),source_input_shas=compiled['bindings'],added=added,monitor_records=monitors,
            training_photos=len(counts),training_exposures=len(listing),training_bottle_boxes=sum(len(experiment.label_path(Path(p)).read_text().splitlines()) for p in counts),
            training_listing_sha=sha(RUN/'training-exposures.txt'),monitor_listing_sha=sha(RUN/'monitor-images.txt'),monitor_photos=len(val),monitor_bottle_boxes=sum(len(r['rows']) for r in monitors),config=config,
            full_training_started=False,independent_NHS_test=False,source_repeats_are_independent_photos=False)
 save(RUN/'dataset.json',audit);verify();print('FULL_DATA_PREPARED',audit['training_photos'],audit['training_bottle_boxes'],audit['training_exposures'],flush=True)
if __name__=='__main__':main()
