"""Separate reviewed training expansion; permanent monitor/benchmark stay fixed."""
from pathlib import Path
from collections import Counter
import hashlib
import importlib.util
import json
import shutil
import yaml
import experiment

RUN=Path(__file__).resolve().parent;BASE=RUN.parent/'monitor-campaign-v34';STAGE=RUN.parent/'data-followup-v35';OI=RUN.parent/'openimages-staging-v32'
CACHE=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-reviewed-neck-monitor-v36')
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
base=module('v36_base',BASE/'build.py');m=base.m;s=base.s
admission=module('v36_admission',RUN/'admission.py');screening=module('v36_screening',STAGE/'screen_followup.py')

def candidates():
 bindings={}
 def bound(path):
  raw=path.read_bytes();bindings[str(path)]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
 checked=bound(STAGE/'overlap-screen.json');assert checked['status']=='screened_only_no_training_admission'
 screening.check_inputs(checked['inputs'])
 assert checked['benchmark_images']==246 and checked['permanent_monitor_images']==36 and checked['protected_unique_photos']==1448
 current=bound(BASE/'dataset.json');manifest=m.MANIFEST
 forbidden_ids={r['id'] for r in current['monitor_records']}
 forbidden_shas={r['sha'] for r in manifest['records']}|{r['image_sha'] for r in current['monitor_records']}
 forbidden_pixels={r['pixels'] for r in manifest['records']}|{r['pixels'] for r in current['monitor_records']}
 positive_rows=bound(STAGE/'additional-downloads.json');positives={r['id']:r for r in positive_rows}
 generic={r['id']:r for r in bound(OI/'downloads.json')}
 positive_first=bound(STAGE/'positive-first-review.json');positive_second=bound(STAGE/'positive-second-review.json')
 positive_ids={positive_rows[i]['id'] for i in positive_second['accepted_indices']}
 assert set(positive_second['accepted_indices'])<=set(positive_first['accepted_indices_pending_second_review_and_overlap'])
 negative_first=bound(STAGE/'negative-first-review.json');negative_second=bound(STAGE/'negative-second-review.json')
 negative_ids={r['id'] for r in negative_first if r['index'] in negative_second['accepted_indices']}
 negative_review_by_id={r['id']:r for r in negative_first}
 target_groups={r['id']:r for r in bound(OI/'targeted-annotation-prefix.json')['groups']}
 target_meta={r['ImageID']:r for r in bound(OI/'targeted-photo-metadata.json')['found']}
 generic_groups={r['id']:r for r in bound(OI/'annotation-prefix.json')['groups']}
 generic_meta={r['ImageID']:r for r in bound(OI/'source-photo-metadata.json')['found']}
 labels={r['id']:r['positive_source_labels'] for r in bound(OI/'plastic-water-image-labels.json')['records']}
 taco_rows=bound(STAGE/'taco-downloads.json');tacos={r['id']:r for r in taco_rows}
 taco_first=bound(STAGE/'taco-first-review.json');taco_second=bound(STAGE/'taco-second-review.json')
 taco_ids={taco_rows[i]['id'] for i in taco_second['accepted_indices']}
 assert set(taco_second['accepted_indices'])<=set(taco_first['accepted_indices_pending_second_review_metadata_and_overlap'])
 taco_metadata={r['id']:r for r in bound(STAGE/'taco-photo-metadata.json')}
 taco_path=RUN.parents[2]/'data/focus-three-class-improvement/bottle-candidates/taco_annotations.json'
 taco_raw=bound(taco_path);taco_images={r['id']:r for r in taco_raw['images']};taco_annotations={}
 for box in taco_raw['annotations']:taco_annotations.setdefault(box['image_id'],[]).append(box)
 added=[];reviews=[]
 for check in checked['records']:
  if check['hold']:continue
  identity=check['id'];role=check['role'];kind=check['source']
  if kind=='openimages':
   assert identity in (positive_ids if role=='positive' else negative_ids)
   row=positives[identity] if role=='positive' else generic[identity]
   groups,metadata=(target_groups,target_meta) if role=='positive' else (generic_groups,generic_meta)
   source=dict(row,source_type=kind,role=role)
   official=dict(annotations=groups[identity]['annotations'],photo_metadata=metadata[identity])
   if role=='positive':
    index=next(i for i,r in enumerate(positive_rows) if r['id']==identity)
    reviewed=positive_first['records'][index];assert reviewed['index']==index and reviewed['id']==identity and reviewed['decision']=='candidate_pending_second_review_and_overlap'
    reviewed_sha=reviewed['sha']
    assert row['positive_source_labels']==groups[identity]['positive_source_labels']==labels[identity]
    assert any(r['ImageID']==identity and r['LabelName'] in {'/m/02rlncx','/m/0118n_9r'} and float(r['Confidence'])==1 for r in labels[identity])
   else:
    reviewed=negative_review_by_id[identity];assert reviewed['image']==row['image'] and reviewed['photo_metadata']==row['photo_metadata']
    reviewed_sha=reviewed['image_sha']
  else:
   assert kind=='taco' and role=='positive' and identity in taco_ids
   row=tacos[identity];original=taco_images[row['original']['id']];annotations=taco_annotations[original['id']]
   index=next(i for i,r in enumerate(taco_rows) if r['id']==identity)
   reviewed=taco_first['records'][index];assert reviewed['index']==index and reviewed['id']==identity and reviewed['decision']=='candidate'
   reviewed_sha=reviewed['sha']
   assert row['source_annotation_sha']==bindings[str(taco_path)]
   source=dict(row,source_type=kind,role=role,width=original['width'],height=original['height'],photo_metadata=taco_metadata[identity])
   official=dict(original=original,annotations=[a for a in annotations if a['category_id']==5],all_source_annotations=annotations,photo_metadata=taco_metadata[identity])
  review=dict(id=identity,accepted=True,full_scene_reviewed=True,bottle_scope_reviewed=True,independent_second_review=True,human_verified=False,
              annotation_indices=list(range(len(source['annotations']))) if role=='positive' else [],source_annotations=source['annotations'],
              image_sha=reviewed_sha,metadata_sha=admission.json_sha(source['photo_metadata']))
  if role=='negative':review.update(verified_absence=True,all_source_boxes_non_target=True)
  record=admission.admit(source,review,official,check,forbidden_ids,forbidden_shas,forbidden_pixels)
  assert record['source_image']==check['image']
  added.append(record);reviews.append(review);forbidden_ids.add(identity);forbidden_shas.add(record['image_sha']);forbidden_pixels.add(record['pixels'])
 screening.check_inputs(bindings)
 return added,reviews,bindings

def verify():
 source=base.verify();audit=s.load(RUN/'dataset.json');added,reviews,bindings=candidates()
 assert audit['base_dataset_audit_sha']==s.sha(BASE/'dataset.json') and audit['source_input_shas']==bindings
 assert audit['compiled_reviews']==reviews
 exposures=Counter((BASE/'training-exposures.txt').read_text().splitlines())
 for expected,record in zip(added,audit['added'],strict=True):
  assert all(record[k]==v for k,v in expected.items())
  assert admission.hashes(record['image'])==(record['image_sha'],record['pixels'])
  assert experiment.label_path(Path(record['image'])).read_text().splitlines()==record['rows']
  exposures[record['image']]+=4
 listing=(RUN/'training-exposures.txt').read_text().splitlines()
 assert Counter(listing)==exposures and s.sha(RUN/'training-exposures.txt')==audit['training_listing_sha']
 assert yaml.safe_load((RUN/'data.yaml').read_text())==audit['config']
 assert audit['monitor_records']==source['monitor_records'] and audit['monitor_fingerprint']==source['monitor_fingerprint']
 assert audit['training_photos']==len(exposures) and audit['training_exposures']==len(listing)
 assert audit['training_bottle_boxes']==sum(len(experiment.label_path(Path(p)).read_text().splitlines()) for p in exposures)
 assert audit['final_benchmark_images']==246 and audit['final_benchmark_bottle_boxes']==67
 return audit

if __name__=='__main__':
 source=base.verify();added,reviews,bindings=candidates();exposures=Counter((BASE/'training-exposures.txt').read_text().splitlines())
 for record in added:
  image=CACHE/'images/train'/Path(record['source_image']).name;label=experiment.label_path(image)
  image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
  if image.exists():assert s.sha(image)==record['image_sha']
  else:shutil.copyfile(record['source_image'],image)
  rows=experiment.validate_rows(record['rows'],image,['bottle']);assert rows==record['rows'];text='\n'.join(rows)+('\n' if rows else '')
  if label.exists():assert label.read_text()==text
  else:label.write_text(text)
  record.update(image=str(image),label=str(label));exposures[str(image)]+=4
 listing=[p for p,n in exposures.items() for _ in range(n)];(RUN/'training-exposures.txt').write_text('\n'.join(listing)+'\n')
 config=dict(path=str(m.DATA),train=str(RUN/'training-exposures.txt'),val=source['config']['val'],names=['bottle']);(RUN/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
 audit=dict(source,base_dataset_audit_sha=s.sha(BASE/'dataset.json'),source_input_shas=bindings,compiled_reviews=reviews,added=added,
            training_photos=len(exposures),training_exposures=len(listing),training_bottle_boxes=sum(len(experiment.label_path(Path(p)).read_text().splitlines()) for p in exposures),
            training_listing_sha=s.sha(RUN/'training-exposures.txt'),config=config)
 s.save(RUN/'dataset.json',audit);verify();print('V36_REVIEWED_EXPANSION_VERIFIED',len(added),len(exposures),audit['training_bottle_boxes'],len(listing),flush=True)
