"""Join reviewed sources to raw provenance and immutable overlap receipts."""
from pathlib import Path
import hashlib
import importlib.util
import json

RUN=Path(__file__).resolve().parent;ROOT=RUN.parent
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
admission=module('v40_source_admission',RUN/'admission.py')
helper=module('v40_overlap_helpers',ROOT/'data-followup-v35/screen_followup.py')

def compile_all(require_negatives=True):
 bindings={}
 def bound(path):
  obj,digest=helper.snapshot(path);bindings[str(path)]=digest;return obj
 positive_screen=bound(ROOT/'openimages-expansion-staging-v37/combined-overlap-screen.json');helper.check_inputs(positive_screen['inputs'])
 positive_checks={r['id']:r for r in positive_screen['records']}
 litter=ROOT/'taco-expansion-staging-v38';source_path=litter/'annotations-unofficial.json';raw=bound(source_path)
 images={r['id']:r for r in raw['images']};annotations={}
 for a in raw['annotations']:annotations.setdefault(a['image_id'],[]).append(a)
 monitor=bound(ROOT/'monitor-campaign-v34/dataset.json')
 forbidden_ids={r['id'] for r in monitor['monitor_records']}
 forbidden_shas={r['image_sha'] for r in monitor['monitor_records']}|{r['sha'] for r in positive_screen['protected_records']}
 forbidden_pixels={r['pixels'] for r in monitor['monitor_records']}
 all_records=[];reviews=[]
 def compile_taco(folder,checks,role):
  rows=bound(folder/'downloads.json');first=bound(folder/'first-review.json');second=bound(folder/'second-review.json')
  assert second['first_review_sha256']==bindings[str(folder/'first-review.json')]
  for index in second['accepted_indices']:
   row=rows[index];checked=checks[row['id']]
   if checked['hold']:continue
   reviewed=first['records'][index]
   assert reviewed['index']==index and reviewed['id']==row['id'] and reviewed['decision']=='candidate' and reviewed['sha']==row['sha']
   original=images[row['original']['id']];boxes=annotations[original['id']];assert row['source_annotation_sha']==bindings[str(source_path)]
   source=dict(row,source_type='taco',role=role,width=original['width'],height=original['height'])
   official=dict(original=original,annotations=[a for a in boxes if a['category_id']==5],all_source_annotations=boxes,photo_metadata=row['photo_metadata'])
   review=dict(id=row['id'],accepted=True,full_scene_reviewed=True,bottle_scope_reviewed=True,independent_second_review=True,human_verified=False,
               annotation_indices=list(range(len(row['annotations']))) if role=='positive' else [],source_annotations=row['annotations'],image_sha=reviewed['sha'],metadata_sha=admission.json_sha(row['photo_metadata']))
   if role=='negative':
    assert reviewed['verified_absence'] is True
    review.update(verified_absence=True,all_source_boxes_non_target=True)
   record=admission.admit(source,review,official,checked,forbidden_ids,forbidden_shas,forbidden_pixels)
   assert record['source_image']==checked['image']
   record.update(width=source['width'],height=source['height'],source_group='flickr/'+row['photo_metadata']['photo_id'],source_ledger=str(folder/'downloads.json'),source_index=index)
   all_records.append(record);reviews.append(review)
 compile_taco(litter,positive_checks,'positive')
 if require_negatives:
  negative_screen=bound(ROOT/'commons-background-staging-v41/negative-overlap-screen.json');helper.check_inputs(negative_screen['inputs'])
  negative_checks={r['id']:r for r in negative_screen['records']}
  compile_taco(ROOT/'litter-background-staging-v39',negative_checks,'negative')
  commons=ROOT/'commons-background-staging-v41';rows=bound(commons/'downloads.json');first=bound(commons/'first-review.json');second=bound(commons/'second-review.json')
  assert second['first_review_sha256']==bindings[str(commons/'first-review.json')]
  snapshots={name:bound(commons/name) for name in ['cigarette-source.json','litter-source.json']}
  for index in second['accepted_indices']:
   row=rows[index];checked=negative_checks[row['id']]
   if checked['hold']:continue
   reviewed=first['records'][index]
   assert reviewed['index']==index and reviewed['id']==row['id'] and reviewed['decision']=='candidate' and reviewed['sha']==row['sha'] and reviewed['verified_absence'] is True
   assert row['source_api_sha']==bindings[str(commons/row['source_api_snapshot'])]
   official_page=snapshots[row['source_api_snapshot']]['query']['pages'][str(row['commons_record']['pageid'])]
   assert row['commons_record']==official_page
   source=dict(row,source_type='commons',role='negative',annotations=[],photo_metadata=official_page)
   official=dict(annotations=[],photo_metadata=official_page)
   review=dict(id=row['id'],accepted=True,full_scene_reviewed=True,bottle_scope_reviewed=True,independent_second_review=True,human_verified=False,
               annotation_indices=[],source_annotations=[],image_sha=reviewed['sha'],metadata_sha=admission.json_sha(official_page),verified_absence=True,all_source_boxes_non_target=True)
   record=admission.admit(source,review,official,checked,forbidden_ids,forbidden_shas,forbidden_pixels)
   assert record['source_image']==checked['image']
   record.update(width=row['width'],height=row['height'],source_group='commons/'+str(official_page['pageid']),source_ledger=str(commons/'downloads.json'),source_index=index)
   all_records.append(record);reviews.append(review)
 for key in ['id','image_sha','pixels','source_group']:
  assert len({r[key] for r in all_records})==len(all_records),'Repeated source/capture identity'
 helper.check_inputs(bindings)
 return dict(records=all_records,compiled_reviews=reviews,source_input_shas=bindings,permanent_monitor_records=monitor['monitor_records'])

if __name__=='__main__':
 result=compile_all(require_negatives=False)
 print('REVIEWED_POSITIVE_SOURCES_VERIFIED',len(result['records']),sum(len(r['rows']) for r in result['records']),flush=True)
