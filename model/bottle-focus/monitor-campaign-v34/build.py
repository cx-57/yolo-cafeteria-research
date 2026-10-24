"""Training-only monitor partition; original benchmark stays separate and frozen."""
from pathlib import Path
from collections import Counter
import importlib.util
import hashlib
import json
import re
import shutil
import yaml
import experiment

RUN=Path(__file__).resolve().parent;BASE=RUN.parent/'reviewed-background-v33'
CACHE=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-monitor-v34')
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
base=module('v34_base',BASE/'build.py');m=base.m;s=base.s
partition=module('v34_partition',RUN/'partition.py')
PROVENANCE=['monitor-source-proposal.json','monitor-overlap-screen.json','historical-overlap-screen.json','screen-resolutions.json']

def sources():
 proposal=s.load(RUN/PROVENANCE[0]);records=proposal['records'];assert len(records)==36
 positives={r['id']:r for r in s.load(BASE/'dataset.json')['added']}
 generic={r['id']:r for r in s.load(RUN.parent/'openimages-staging-v32/downloads.json')}
 for r in records:
  assert r['independent_second_review'] is True and r['human_verified'] is False
  if r['role']=='positive':
   original=positives[r['id']];assert r['rows']==original['rows'] and r['image_sha']==original['image_sha']
  else:
   assert r['role']=='reviewed_negative' and not r['rows'] and r['image_sha']==generic[r['id']]['sha']
 lookup={r['id']:r for r in records};resolutions=s.load(RUN/'screen-resolutions.json');allowed={(r['id'],r['training_name']):r for r in resolutions}
 seen=set()
 for name in ['monitor-overlap-screen.json','historical-overlap-screen.json']:
  screen=s.load(RUN/name);assert screen['proposal_sha256']==s.sha(RUN/'monitor-source-proposal.json') and len(screen['records'])==36
  for r in screen['records']:
   assert r['image_sha']==lookup[r['id']]['image_sha'] and not r['benchmark_hold']
   assert not r['exact_matches'] and not r['feature_matches']
   for training_name in r['related_training_photos']:
    resolution=allowed[(r['id'],training_name)]
    assert resolution['resolution']=='visually_different_photos' and resolution['image_sha']==r['image_sha']
    assert resolution['training_sha']==s.sha(m.DATA/'images/train'/training_name)
    seen.add((r['id'],training_name))
 assert seen==set(allowed)
 return records

def groups_for(exposures):
 original={r['name']:r for r in m.MANIFEST['records']};groups={}
 for p in exposures:
  name=Path(p).name;oo=re.search(r'openimages__train__([0-9a-f]{16})',name);lvis=re.search(r'lvis_(\d+)',name)
  if oo:group='openimages/'+oo.group(1)
  elif lvis:group='coco/'+str(int(lvis.group(1)))
  elif name in original:group=str(original[name].get('capture') or original[name]['group'])
  else:group='source/'+name
  groups[p]=group
 return groups

def training_partition(records):
 original=Counter((BASE/'training-exposures.txt').read_text().splitlines())
 return partition.split(original,groups_for(original),records)

def fingerprint(records):
 core=[{k:r[k] for k in ['id','image_sha','pixels','rows','source_group']} for r in records]
 return hashlib.sha256(json.dumps(core,sort_keys=True).encode()).hexdigest()

def verify():
 base.verify();audit=s.load(RUN/'dataset.json');records=sources();exposures,held=training_partition(records)
 assert audit['base_dataset_audit_sha']==s.sha(BASE/'dataset.json')
 assert audit['provenance_shas']=={name:s.sha(RUN/name) for name in PROVENANCE}
 assert audit['monitor_fingerprint']==fingerprint(records)
 assert audit['held_training_photos']==held
 listing=(RUN/'training-exposures.txt').read_text().splitlines();assert Counter(listing)==exposures and s.sha(RUN/'training-exposures.txt')==audit['training_listing_sha']
 assert yaml.safe_load((RUN/'data.yaml').read_text())==audit['config']
 for source,record in zip(records,audit['monitor_records'],strict=True):
  assert all(record[k]==v for k,v in source.items())
  assert partition.hashes(record['image'])==(record['image_sha'],record['pixels'])
  assert Path(record['label']).read_text().splitlines()==record['rows']
 assert {p.name for p in (CACHE/'images/val').glob('*.jpg')}=={Path(r['image']).name for r in audit['monitor_records']}
 assert len(exposures)==audit['training_photos']==567 and len(listing)==audit['training_exposures']==2100
 boxes=sum(len(experiment.label_path(Path(p)).read_text().splitlines()) for p in exposures)
 assert boxes==audit['training_bottle_boxes']==521
 assert audit['monitor_images']==36 and audit['monitor_bottle_boxes']==17
 assert audit['final_benchmark_images']==246 and audit['final_benchmark_bottle_boxes']==67
 return audit

if __name__=='__main__':
 source=base.verify();records=sources();exposures,held=training_partition(records);compiled=[]
 for r in records:
  image=CACHE/'images/val'/Path(r['source_image']).name;label=CACHE/'labels/val'/f'{image.stem}.txt';image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
  if image.exists():assert s.sha(image)==r['image_sha']
  else:shutil.copyfile(r['source_image'],image)
  rows=experiment.validate_rows(r['rows'],image,['bottle']);assert rows==r['rows'];text='\n'.join(rows)+('\n' if rows else '')
  if label.exists():assert label.read_text()==text
  else:label.write_text(text)
  compiled.append(dict(r,image=str(image),label=str(label)))
 listing=[p for p,n in exposures.items() for _ in range(n)];(RUN/'training-exposures.txt').write_text('\n'.join(listing)+'\n')
 config=dict(path=str(m.DATA),train=str(RUN/'training-exposures.txt'),val=str(CACHE/'images/val'),names=['bottle']);(RUN/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
 audit=dict(base_dataset_audit_sha=s.sha(BASE/'dataset.json'),provenance_shas={name:s.sha(RUN/name) for name in PROVENANCE},monitor_fingerprint=fingerprint(records),monitor_records=compiled,held_training_photos=held,training_photos=len(exposures),training_exposures=len(listing),training_bottle_boxes=521,training_listing_sha=s.sha(RUN/'training-exposures.txt'),monitor_images=36,monitor_positive_photos=12,monitor_negative_photos=24,monitor_bottle_boxes=17,final_benchmark_images=246,final_benchmark_bottle_boxes=67,fixed_source_fingerprint=m.MANIFEST['source_fingerprint'],validation_protocol=m.MANIFEST['validation_protocol'],config=config,independent_NHS_test=False,limitations=source['review']+'; continuation monitor not independent NHS test; possible generic pretraining exposure; source overlap screen is heuristic; no model-score-based source selection')
 s.save(RUN/'dataset.json',audit);verify();print('MONITORED_DATA_VERIFIED',len(exposures),len(listing),36,17,flush=True)
