"""Join immutable review decisions to official source annotations and metadata."""
from pathlib import Path
import importlib.util
import json
import re
RUN=Path(__file__).resolve().parent;ROOT=RUN.parent
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
v40=module('v42_prior_sources',ROOT/'source-coverage-diagnostic-v40/compile_sources.py')
admission=v40.admission;helper=v40.helper
partition=module('v42_partition',RUN/'partition.py')
def group(record):
 if 'source_group' in record:return record['source_group']
 meta=record['photo_metadata']
 if record['source_type']=='openimages':
  match=re.search(r'/photos/[^/]+/(\d+)',meta['OriginalLandingURL'])
  assert match,'Missing source capture identity'
  return 'flickr/'+match.group(1)
 return 'flickr/'+meta['photo_id']
def compile_all():
 prior=v40.compile_all();bindings=dict(prior['source_input_shas'])
 def bound(path):
  obj,digest=helper.snapshot(path);bindings[str(path)]=digest;return obj
 diagnostic=bound(ROOT/'source-coverage-diagnostic-v40/dataset.json')
 assert diagnostic['source_input_shas']==prior['source_input_shas']
 selected={r['id'] for r in diagnostic['selection']['monitor']}
 records=[r for r in prior['records'] if r['id'] not in selected]
 assert records==sorted(diagnostic['selection']['remaining'],key=lambda r:prior['records'].index(r))
 old_photo_rows=bound(ROOT/'openimages-staging-v32/downloads.json')+bound(ROOT/'openimages-staging-v32/targeted-downloads.json')
 old_photos={r['id']:r for r in old_photo_rows}
 monitors=[dict(r,original_source_group=r['source_group'],source_group=group(dict(old_photos[r['id']],source_type='openimages'))) for r in prior['permanent_monitor_records']]+diagnostic['records']
 stage=ROOT/'openimages-expansion-staging-v37'
 checks=bound(stage/'combined-overlap-screen.json');helper.check_inputs(checks['inputs']);checks={r['id']:r for r in checks['records']}
 rows=bound(stage/'downloads.json');first=bound(stage/'first-review.json');second=bound(stage/'second-review.json')
 assert second['first_review_sha256']==bindings[str(stage/'first-review.json')]
 groups_doc=bound(stage/'novel-complete-boxes.json');labels_path=stage/'novel-plastic-water-labels.json';labels_doc=bound(labels_path)
 assert groups_doc['image_label_filter_sha256']==bindings[str(labels_path)]
 groups={r['id']:r for r in groups_doc['groups']};labels={r['id']:r['positive_source_labels'] for r in labels_doc['records']}
 metadata={r['ImageID']:r for r in bound(stage/'novel-photo-metadata.json')['found']}
 for index in second['accepted_indices']:
  row=rows[index];checked=checks[row['id']]
  if checked['hold']:continue
  reviewed=first['records'][index]
  assert reviewed['index']==index and reviewed['id']==row['id'] and reviewed['decision']=='candidate' and reviewed['sha']==row['sha']
  assert row['positive_source_labels']==groups[row['id']]['positive_source_labels']==labels[row['id']]
  assert any(a['ImageID']==row['id'] and a['LabelName'] in {'/m/02rlncx','/m/0118n_9r'} and float(a['Confidence'])==1 for a in labels[row['id']])
  source=dict(row,source_type='openimages',role='positive')
  official=dict(annotations=groups[row['id']]['annotations'],photo_metadata=metadata[row['id']])
  review=dict(id=row['id'],accepted=True,full_scene_reviewed=True,bottle_scope_reviewed=True,independent_second_review=True,human_verified=False,
              annotation_indices=list(range(len(row['annotations']))),source_annotations=row['annotations'],image_sha=reviewed['sha'],metadata_sha=admission.json_sha(row['photo_metadata']))
  record=admission.admit(source,review,official,checked)
  record.update(width=row['width'],height=row['height'],source_group=group(record),source_ledger=str(stage/'downloads.json'),source_index=index)
  records.append(record)
 partition.check(records,monitors);helper.check_inputs(bindings)
 return dict(records=sorted(records,key=lambda r:r['id']),bindings=bindings,monitor_records=monitors)
