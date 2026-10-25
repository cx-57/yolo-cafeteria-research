"""Reuse frozen V44parents/monitors; quarantine bottle-like negatives and reweight."""
from pathlib import Path
import importlib.util
import sys
import yaml
import weights
RUN=Path(__file__).resolve().parent;ROOT=RUN.parent;V44=ROOT/'classifier-retention-v44'
sys.path.insert(0,str(V44))
try:
 spec=importlib.util.spec_from_file_location('v45_prior_data',V44/'data.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
finally:sys.path.pop(0)
parent=prior.parent;sha=prior.sha;save=prior.save;load=prior.load;helper=prior.helper
CACHE=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-hardnegative-logits-v45')
def derive():
 base=prior.verify();review=load(RUN/'hard-review.json');mining=load(RUN/'mining.json');candidates=load(RUN/'hard-candidates.json')
 assert review['mining_sha']==sha(RUN/'mining.json') and review['candidate_sha']==sha(RUN/'hard-candidates.json')
 assert review['independent_second_review'] is True and review['human_verified'] is False and review['full_scene_reviewed'] is True
 assert mining['source_dataset_sha']==sha(V44/'dataset.json') and mining['source_weights_sha']=='88877bbf13c02a202811b6ca0a1ccdc2bde70b2916d366f4b67938df732d1518'
 weights.validate_mining(mining['records'],candidates,base['training_records'])
 assert mining['source']=='training_only_no_monitoring_or_benchmark_inference' and mining['mining_confidence']==.1 and mining['nms_iou']==.5 and mining['imgsz']==640 and mining['batch']==4
 parent_by_id={r['id']:r for r in base['training_records']};approved=set()
 for index in review['accepted_indices']:
  c=candidates[index];r=review['records'][index];p=parent_by_id[c['id']]
  assert r['index']==index and r['id']==c['id']==p['id'] and r['image_sha']==c['image_sha']==p['image_sha']
  assert r['verified_absence'] is True and r['decision']=='upsample' and p['rows']==c['rows']==[] and c['predictions']
  assert Path(p['label']).read_text().splitlines()==[];approved.add(c['id'])
 scope=load(RUN/'user-scope.json');assert scope['target']=='disposable_plastic_drink_bottles' and review['user_scope_sha']==sha(RUN/'user-scope.json')
 first=load(RUN/'legacy-first-review.json');second=load(RUN/'legacy-second-review.json');legacy=load(RUN/'legacy-scope-records.json')
 assert first['source_record_sha']==sha(RUN/'legacy-scope-records.json') and first['source_dataset_sha']==legacy['source_dataset_sha']==sha(V44/'dataset.json')
 assert second['first_review_sha']==sha(RUN/'legacy-first-review.json')
 expected_legacy=sorted([r for r in base['training_records'] if r['rows'] and ('lvis_' in r['id'] or r['id'].startswith('external_bottle__'))],key=lambda r:r['id'])
 assert legacy['records']==expected_legacy and {r['id'] for r in legacy['records']}=={r['id'] for r in first['records']}
 held=weights.legacy_holds(base['training_records'],first,second)+[parent_by_id[candidates[i]['id']] for i in review['held_indices']]
 old=load(V44/'source-selection.json')['old_monitor_records'];monitors=old+base['retention_records']
 records=weights.reweight(base['training_records'],monitors,held,approved,review['extra_exposures'])
 return base,records,held

def verify():
 base,records,held=derive();audit=load(RUN/'dataset.json')
 assert audit['training_records']==records and audit['retention_records']==base['retention_records'] and audit['retention_fingerprint']==base['retention_fingerprint']
 assert audit['source_input_shas']=={str(p):sha(p) for p in [V44/'dataset.json',V44/'source-selection.json',RUN/'mining.json',RUN/'hard-candidates.json',RUN/'hard-review.json',RUN/'user-scope.json',RUN/'legacy-scope-records.json',RUN/'legacy-first-review.json',RUN/'legacy-second-review.json']}
 listing=[r['image'] for r in records for _ in range(r['exposures'])]
 assert (RUN/'training-exposures.txt').read_text().splitlines()==listing and audit['training_listing_sha']==sha(RUN/'training-exposures.txt')
 assert (RUN/'retention-images.txt').read_text().splitlines()==[r['image'] for r in base['retention_records']]
 assert yaml.safe_load((RUN/'data.yaml').read_text())==audit['config']
 assert audit['training_photos']==len(records) and audit['training_boxes']==sum(len(r['rows']) for r in records) and audit['training_exposures']==len(listing)
 return audit

def main():
 assert not (RUN/'measurement-start.json').exists()
 base,records,held=derive();listing=[r['image'] for r in records for _ in range(r['exposures'])]
 (RUN/'training-exposures.txt').write_text('\n'.join(listing)+'\n');(RUN/'retention-images.txt').write_text('\n'.join(r['image'] for r in base['retention_records'])+'\n')
 config=dict(path=str(CACHE),train=str(RUN/'training-exposures.txt'),val=str(RUN/'retention-images.txt'),names=['bottle']);(RUN/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
 audit=dict(training_records=records,retention_records=base['retention_records'],retention_fingerprint=base['retention_fingerprint'],retention_photos=24,retention_boxes=16,training_photos=len(records),training_boxes=sum(len(r['rows']) for r in records),training_exposures=len(listing),
            training_listing_sha=sha(RUN/'training-exposures.txt'),config=config,whole_group_quality_holds=sorted({weights.canonical_group(r) for r in held}),
            source_input_shas={str(p):sha(p) for p in [V44/'dataset.json',V44/'source-selection.json',RUN/'mining.json',RUN/'hard-candidates.json',RUN/'hard-review.json',RUN/'user-scope.json',RUN/'legacy-scope-records.json',RUN/'legacy-first-review.json',RUN/'legacy-second-review.json']},
            no_new_training_parent_images=True,no_label_edits=True,retention_previously_seen_by_initial_and_v42=True,independent_NHS_test=False)
 save(RUN/'dataset.json',audit);verify();print('LOGIT_DATA_READY',audit['training_photos'],audit['training_boxes'],audit['training_exposures'],flush=True)
if __name__=='__main__':main()
