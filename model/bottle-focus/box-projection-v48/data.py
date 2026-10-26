"""Add twice-reviewed nonoverlapping source parents to immutable V45 training."""
from pathlib import Path
import importlib.util
import sys
import shutil
import yaml
RUN=Path(__file__).resolve().parent;ROOT=RUN.parent;V45=ROOT/'hard-negative-logits-v45';STAGE=ROOT/'openimages-drink-staging-v47'
sys.path.insert(0,str(V45))
try:
 spec=importlib.util.spec_from_file_location('v48_prior_data',V45/'data.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
finally:sys.path.pop(0)
parent=prior.parent;sha=prior.sha;load=prior.load;save=prior.save
CACHE=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-box-projection-v48')
def derive():
 base=prior.verify();screen=load(STAGE/'overlap-screen.json');parent.sources.helper.check_inputs(screen['inputs'])
 assert screen['permanent_monitor_photos']==92 and screen['original_benchmark_photos']==246
 first=load(STAGE/'first-review.json');second=load(STAGE/'second-review-independent.json');photos=load(STAGE/'downloads.json')
 assert first['source_downloads_sha']==second['source_downloads_sha']==sha(STAGE/'downloads.json')
 assert second['independent_review'] is True and first['target_scope']==second['target_scope']=='disposable_plastic_drink_bottles'
 assert len(first['records'])==len(second['records'])==len(photos)==80
 groups={r['id']:r for r in load(ROOT/'openimages-expansion-staging-v37/novel-complete-boxes.json')['groups']}
 metadata={r['ImageID']:r for r in load(ROOT/'openimages-expansion-staging-v37/novel-photo-metadata.json')['found']}
 selection=load(STAGE/'source-selection.json');assert selection['no_model_outputs_used'] is True
 parent.sources.helper.check_inputs(selection['source_input_shas'])
 old=load(ROOT/'classifier-retention-v44/source-selection.json')['old_monitor_records'];monitors=base['retention_records']+old
 blocked_groups={prior.weights.canonical_group(r) for r in base['training_records']+monitors}
 blocked_bytes={r['image_sha'] for r in base['training_records']+monitors};blocked_pixels={r['pixels'] for r in base['training_records']+monitors}
 added=[];group_holds=[]
 for result in screen['records']:
  i=result['index'];p=photos[i];a=first['records'][i];b=second['records'][i]
  assert a['index']==b['index']==i and a['id']==b['id']==p['id']==result['id']
  assert a['sha']==b['image_sha']==p['sha']==result['sha'] and a['decision']==b['decision']=='candidate'
  assert p['annotations']==groups[p['id']]['annotations'] and p['photo_metadata']==metadata[p['id']]
  if result['hold']:continue
  assert all(not result[k] for k in ['byte_matches','pixel_matches','dhash_matches','feature_matches'])
  assert p['official_split']=='train' and p['photo_metadata']['Rotation'] in ['0','0.0'] and p['photo_metadata']['License'].startswith('https://creativecommons.org/licenses/')
  assert any(r['LabelName'] in ['/m/02rlncx','/m/0118n_9r'] and float(r['Confidence'])==1 for r in p['positive_source_labels'])
  group=parent.sources.group(dict(p,source_type='openimages'))
  if group in blocked_groups:group_holds.append(p['id']);continue
  assert p['sha'] not in blocked_bytes and p['pixels'] not in blocked_pixels
  from PIL import Image
  with Image.open(p['image']) as picture:assert picture.size==(p['width'],p['height']) and picture.getexif().get(274,1)==1
  assert sha(p['image'])==p['sha']
  rows=[]
  for box in p['annotations']:
   assert box['LabelName']=='/m/04dr76w' and box['IsGroupOf']==box['IsDepiction']=='0'
   x,z,y,t=[float(box[k]) for k in ['XMin','XMax','YMin','YMax']];assert 0<=x<z<=1 and 0<=y<t<=1
   rows.append('0 '+' '.join(f'{v:.9f}' for v in [(x+z)/2,(y+t)/2,z-x,t-y]))
  name=f"openimages__train__{p['id']}";image=CACHE/'images/train'/(name+'.jpg');label=CACHE/'labels/train'/(name+'.txt')
  added.append(dict(id=name,source_image=p['image'],image=str(image),label=str(label),image_sha=p['sha'],pixels=p['pixels'],source_group=group,rows=rows,exposures=4,source_index=i,source_type='openimages',source_ledger=str(STAGE/'downloads.json'),human_verified=False,independent_second_review=True))
  blocked_groups.add(group);blocked_bytes.add(p['sha']);blocked_pixels.add(p['pixels'])
 inputs={str(p):sha(p) for p in [V45/'dataset.json',V45/'declaration.json',STAGE/'downloads.json',STAGE/'source-selection.json',STAGE/'first-review.json',STAGE/'second-review-independent.json',STAGE/'overlap-screen.json',ROOT/'openimages-expansion-staging-v37/novel-complete-boxes.json',ROOT/'openimages-expansion-staging-v37/novel-photo-metadata.json']}
 return base,added,group_holds,inputs

def verify():
 base,added,held,inputs=derive();audit=load(RUN/'dataset.json');records=base['training_records']+added
 assert audit['training_records']==records and audit['added']==added and audit['source_input_shas']==inputs and audit['source_group_holds']==held
 assert audit['retention_records']==base['retention_records'] and audit['retention_fingerprint']==base['retention_fingerprint']
 for r in records+base['retention_records']:
  assert sha(r['image'])==r['image_sha'] and Path(r['label']).read_text().splitlines()==r['rows']
 listing=[r['image'] for r in records for _ in range(r['exposures'])]
 assert (RUN/'training-exposures.txt').read_text().splitlines()==listing and audit['training_listing_sha']==sha(RUN/'training-exposures.txt')
 assert (RUN/'retention-images.txt').read_text().splitlines()==[r['image'] for r in base['retention_records']]
 assert yaml.safe_load((RUN/'data.yaml').read_text())==audit['config']
 assert audit['training_photos']==len(records) and audit['training_boxes']==sum(len(r['rows']) for r in records) and audit['training_exposures']==len(listing)
 return audit

def main():
 assert not (RUN/'measurement-start.json').exists();base,added,held,inputs=derive()
 for r in added:
  image=Path(r['image']);label=Path(r['label']);image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
  if not image.exists():shutil.copyfile(r['source_image'],image)
  if label.exists():assert label.read_text().splitlines()==r['rows']
  else:label.write_text('\n'.join(r['rows'])+'\n')
 records=base['training_records']+added;listing=[r['image'] for r in records for _ in range(r['exposures'])]
 (RUN/'training-exposures.txt').write_text('\n'.join(listing)+'\n');(RUN/'retention-images.txt').write_text('\n'.join(r['image'] for r in base['retention_records'])+'\n')
 config=dict(path=str(CACHE),train=str(RUN/'training-exposures.txt'),val=str(RUN/'retention-images.txt'),names=['bottle']);(RUN/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
 audit=dict(training_records=records,added=added,source_group_holds=held,source_input_shas=inputs,retention_records=base['retention_records'],retention_fingerprint=base['retention_fingerprint'],retention_photos=24,retention_boxes=16,training_photos=len(records),training_boxes=sum(len(r['rows']) for r in records),training_exposures=len(listing),training_listing_sha=sha(RUN/'training-exposures.txt'),config=config,no_original_label_edits=True,independent_NHS_test=False)
 save(RUN/'dataset.json',audit);verify();print('BOX_PROJECTION_DATA',len(added),audit['training_photos'],audit['training_boxes'],audit['training_exposures'],flush=True)
if __name__=='__main__':main()
