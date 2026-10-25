"""Immutable reviewed training derivative and a permanently reserved retention cohort."""
from pathlib import Path
from collections import Counter
import importlib.util
import json
import random
import re
import sys
import yaml
import partition
RUN=Path(__file__).resolve().parent;ROOT=RUN.parent;V42=ROOT/'reviewed-full-training-v42';REVIEW=ROOT/'optimization-diagnostic-v43'
CACHE=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-classifier-retention-v44')
sys.path.insert(0,str(V42))
try:import build as parent
finally:sys.path.pop(0)
sha=parent.sha;save=parent.save;helper=parent.sources.helper
load=lambda p:json.loads(Path(p).read_text())
def label(image):
 p=Path(image);return p.parent.parent.parent/'labels'/p.parent.name/(p.stem+'.txt')
def fingerprint(records):
 return parent.sources.admission.json_sha([{k:r[k] for k in ['id','image_sha','pixels','source_group','rows']} for r in records])
def all_sources():
 audit=parent.verify();bindings={}
 def bound(path):
  d,s=helper.snapshot(path);bindings[str(path)]=s;return d
 bound(V42/'dataset.json');bound(V42/'declaration.json');bindings[str(V42/'training-exposures.txt')]=sha(V42/'training-exposures.txt')
 manifest=bound(parent.v36.m.DATA/'manifest.json');original={r['name']:r for r in manifest['records']};first=bound(REVIEW/'training-first-review.json');second=bound(REVIEW/'training-second-review.json')
 assert second['first_review_sha']==bindings[str(REVIEW/'training-first-review.json')]
 source_review=bound(REVIEW/'trash-training-review-records.json');assert first['source_record_sha']==bindings[str(REVIEW/'trash-training-review-records.json')]
 assert len(source_review)==len(first['records'])==88
 reviews={Path(r['image']).name:r for r in first['records']}
 added={Path(r['source_image']).name:r for r in audit['added']}
 counts=Counter((V42/'training-exposures.txt').read_text().splitlines());records=[];held=[]
 for path,n in counts.items():
  image=Path(path);rows=label(image).read_text().splitlines();f=helper.features(image)
  bindings[str(image)]=f['sha'];bindings[str(label(image))]=sha(label(image))
  if image.name in original:group=str(original[image.name].get('capture') or original[image.name]['group'])
  elif image.name in added:group=added[image.name]['source_group']
  else:group=parent.v36.base.groups_for([str(image)])[str(image)]
  record=dict(id=image.stem,source_image=str(image),source_label=str(label(image)),image_sha=f['sha'],pixels=f['pixels'],source_group=group,rows=rows,source_rows=rows,exposures=n)
  if image.name.startswith('trash_detection__'):
   reviewed=reviews[image.name];assert reviewed['image']==str(image) and reviewed['image_sha']==f['sha'] and reviewed['rows']==rows
   if reviewed['index'] not in second['accepted_indices']:held.append(record);continue
   record['rows']=partition.reviewed_rows(record,reviewed,second);record['review_index']=reviewed['index']
  records.append(record)
 negatives=bound(ROOT/'reviewed-background-v33/negative-reviews.json')
 negative_names={r['name'] for r in negatives if r['accepted'] and r['full_scene_reviewed'] and r['independent_second_review']}
 for record in records:
  if not record['rows'] and Path(record['source_image']).name in negative_names:
   review=next(r for r in negatives if r['name']==Path(record['source_image']).name)
   assert review['image_sha']==record['image_sha'] and review['label_sha']==sha(record['source_label'])
   record['reviewed_old_negative']=True
 helper.check_inputs(bindings)
 return records,held,audit['monitor_records'],manifest,bindings

def prepare():
 assert not (RUN/'measurement-start.json').exists(),'Retention selection locked after measurements begin'
 records,held,old_monitors,manifest,bindings=all_sources();held_groups={r['source_group'] for r in held}
 available=partition.keep(records,old_monitors,held_groups,set());cache={}
 def features(record):
  if record['image_sha'] not in cache:cache[record['image_sha']]=helper.protected_features(record['source_image'],record['image_sha'])
  return cache[record['image_sha']]
 protected=[]
 for record in manifest['records']:
  if record['split']=='val':
   path=parent.v36.m.DATA/'images/val'/record['name'];protected.append(('original-benchmark/'+record['name'],helper.protected_features(path,record['sha'])));bindings[str(path)]=record['sha']
 for record in old_monitors:
  path=record['image'];protected.append(('old-monitor/'+record['id'],helper.protected_features(path,record['image_sha'])));bindings[str(path)]=record['image_sha']
 rng=random.Random(20261025)
 positive=[r for r in available if r.get('review_index') is not None and r['rows']]
 negative=[r for r in available if not r['rows'] and (r.get('review_index') in [5,38,77] or r.get('reviewed_old_negative'))]
 # Prefer the three reviewed same-source negatives; choose all other groups with a fixed seed.
 positive.sort(key=lambda r:r['id']);rng.shuffle(positive)
 same=[r for r in negative if r.get('review_index') in [5,38,77]];other=sorted([r for r in negative if r not in same],key=lambda r:r['id']);rng.shuffle(other);negative=sorted(same,key=lambda r:r['id'])+other
 chosen=[];checks=[];seen=set();candidate_holds=set()
 for pool,required in [(positive,16),(negative,8)]:
  accepted=0
  for record in pool:
   if record['source_group'] in seen:continue
   check=helper.compare(features(record),protected+[(r['id'],features(r)) for r in chosen]);checks.append(dict(id=record['id'],**check))
   if check['hold']:candidate_holds.add(record['id']);continue
   chosen.append(record);seen.add(record['source_group']);accepted+=1
   if accepted==required:break
  assert accepted==required,'Insufficient independently grouped source candidates'
 targets=[(r['id'],features(r)) for r in chosen];remaining=partition.keep(available,chosen,held_groups,candidate_holds)
 training=[];aliases=[]
 for record in remaining:
  check=helper.compare(features(record),targets)
  if check['hold']:aliases.append(dict(id=record['id'],source_group=record['source_group'],**check))
  else:training.append(record)
 flag_ids=candidate_holds|{r['id'] for r in aliases}
 extra_held_groups={r['source_group'] for r in records if r['id'] in flag_ids}
 training=partition.keep(available,chosen,held_groups|extra_held_groups,flag_ids)
 helper.check_inputs(bindings)
 save(RUN/'source-selection.json',dict(source_records=records,quality_holds=held,old_monitor_records=old_monitors,retention_records=chosen,training_records=training,
      seed=20261025,candidate_screen=checks,held_training_aliases=aliases,source_input_shas=bindings,retention_previously_seen_by_initial_and_v42=True,independent_NHS_test=False))
 publish()

def publish():
 selection=load(RUN/'source-selection.json');helper.check_inputs(selection['source_input_shas']);training=[];monitor=[]
 for role,records in [('train',selection['training_records']),('val',selection['retention_records'])]:
  for source in records:
   if role=='train' and source['rows']==source['source_rows']:image=Path(source['source_image']);destination=label(image)
   else:
    image=CACHE/'images'/role/Path(source['source_image']).name;destination=label(image);parent.clone(source['source_image'],image);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text('\n'.join(source['rows'])+('\n' if source['rows'] else ''))
   record=dict(source,image=str(image),label=str(destination))
   (training if role=='train' else monitor).append(record)
 listing=[r['image'] for r in training for _ in range(r['exposures'])];(RUN/'training-exposures.txt').write_text('\n'.join(listing)+'\n');(RUN/'retention-images.txt').write_text('\n'.join(r['image'] for r in monitor)+'\n')
 config=dict(path=str(CACHE),train=str(RUN/'training-exposures.txt'),val=str(RUN/'retention-images.txt'),names=['bottle']);(RUN/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
 audit=dict(training_records=training,retention_records=monitor,training_photos=len(training),training_boxes=sum(len(r['rows']) for r in training),training_exposures=len(listing),retention_photos=len(monitor),retention_boxes=sum(len(r['rows']) for r in monitor),
            retention_fingerprint=fingerprint(monitor),source_selection_sha=sha(RUN/'source-selection.json'),training_listing_sha=sha(RUN/'training-exposures.txt'),retention_listing_sha=sha(RUN/'retention-images.txt'),config=config,independent_NHS_test=False,retention_previously_seen_by_initial_and_v42=True)
 save(RUN/'dataset.json',audit);verify();print('CLASSIFIER_DATA_PREPARED',audit['training_photos'],audit['training_boxes'],audit['training_exposures'],audit['retention_photos'],audit['retention_boxes'],flush=True)

def verify():
 audit=load(RUN/'dataset.json');selection=load(RUN/'source-selection.json');assert sha(RUN/'source-selection.json')==audit['source_selection_sha'];helper.check_inputs(selection['source_input_shas'])
 flag_ids={r['id'] for r in selection['held_training_aliases']}|{r['id'] for r in selection['candidate_screen'] if r['hold']}
 held_groups={r['source_group'] for r in selection['quality_holds']}|{r['source_group'] for r in selection['source_records'] if r['id'] in flag_ids};reserved=selection['old_monitor_records']+selection['retention_records']
 expected=partition.keep(selection['source_records'],reserved,held_groups,flag_ids)
 assert expected==selection['training_records']
 for originals,records in [(selection['training_records'],audit['training_records']),(selection['retention_records'],audit['retention_records'])]:
  for original,record in zip(originals,records,strict=True):
   assert all(record[k]==v for k,v in original.items())
   assert parent.sources.admission.hashes(record['image'])==(record['image_sha'],record['pixels'])
   assert Path(record['label']).read_text().splitlines()==record['rows']
 listing=[r['image'] for r in audit['training_records'] for _ in range(r['exposures'])]
 assert (RUN/'training-exposures.txt').read_text().splitlines()==listing and sha(RUN/'training-exposures.txt')==audit['training_listing_sha']
 assert (RUN/'retention-images.txt').read_text().splitlines()==[r['image'] for r in audit['retention_records']] and sha(RUN/'retention-images.txt')==audit['retention_listing_sha']
 assert len(audit['retention_records'])==24 and sum(bool(r['rows']) for r in audit['retention_records'])==16
 assert audit['retention_fingerprint']==fingerprint(audit['retention_records']) and yaml.safe_load((RUN/'data.yaml').read_text())==audit['config']
 assert audit['training_photos']==len(audit['training_records']) and audit['training_boxes']==sum(len(r['rows']) for r in audit['training_records']) and audit['training_exposures']==len(listing)
 return audit
if __name__=='__main__':prepare()
