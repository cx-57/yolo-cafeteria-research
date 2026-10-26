"""Conservative overlap screen of reviewed source photos; no dataset admission."""
from pathlib import Path
import hashlib
import importlib.util
import json
RUN=Path(__file__).resolve().parent;ROOT=RUN.parent
HELPER=ROOT/'data-followup-v35/screen_followup.py'
spec=importlib.util.spec_from_file_location('v47_screen_helper',HELPER);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
BASE=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-capture-validation')
def main():
 identities={str(HELPER):hashlib.sha256(HELPER.read_bytes()).hexdigest(),str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 def bound(path):
  value,digest=helper.snapshot(path);identities[str(path)]=digest;return value
 manifest=bound(BASE/'manifest.json');frozen=bound(ROOT/'validation-review/frozen-files-verification.json');assert identities[str(BASE/'manifest.json')]==frozen['manifest_sha']
 prior=bound(ROOT/'classifier-retention-v44/source-selection.json');current=bound(ROOT/'box-projection-v48/dataset.json')
 rows=bound(RUN/'downloads.json');first=bound(RUN/'first-review.json');assert first['source_downloads_sha']==identities[str(RUN/'downloads.json')]
 second=bound(RUN/'second-review-independent.json');followup=bound(RUN/'followup-review.json');geometry=bound(RUN/'geometry-audit.json');assert followup['prior_independent_review_sha']==identities[str(RUN/'second-review-independent.json')] and followup['source_downloads_sha']==geometry['source_downloads_sha']==identities[str(RUN/'downloads.json')];assert second['source_downloads_sha']==identities[str(RUN/'downloads.json')] and second['independent_review'] is True
 # Review schemas deliberately join actual per-photo SHA and decision, not only indices.
 accepted=[]
 for i,r in enumerate(rows):
  a=next(x for x in first['records'] if x['index']==i);b=next(x for x in second['records'] if x['index']==i)
  assert a['id']==b['id']==r['id']
  if r['status']!='unreviewed_staged_only':continue
  assert a['sha']==b['image_sha']==r['sha']
  if a['decision']==b['decision']=='candidate' and i in followup['revised_independent_candidate_indices']:
   assert geometry['records'][i]['accepted_geometry'] is True;accepted.append(i)
 targets=[dict(id=('benchmark/' if r['split']=='val' else 'historical/')+r['name'],image=str(BASE/'images'/r['split']/r['name']),sha=r['sha']) for r in manifest['records']]
 for label,records in [('source',prior['source_records']),('old-monitor',prior['old_monitor_records']),('retention',prior['retention_records']),('current',current['training_records'])]:
  targets.extend(dict(id=label+'/'+r['id'],image=r.get('image',r['source_image']),sha=r['image_sha']) for r in records)
 prepared=[];seen=set();protected=[]
 for r in targets:
  if r['sha'] in seen:continue
  seen.add(r['sha']);f=helper.protected_features(Path(r['image']),r['sha']);identities[r['image']]=r['sha'];protected.append(r);prepared.append((r['id'],f))
 print('PROTECTED_FEATURES_READY',len(prepared),flush=True)
 results=[]
 for i in accepted:
  r=rows[i];f=helper.features(r['image']);assert f['sha']==r['sha'];identities[r['image']]=r['sha']
  result=dict(index=i,id=r['id'],image=r['image'],sha=r['sha'],pixels=f['pixels'],**helper.compare(f,prepared));results.append(result)
  prepared.append(('candidate/'+r['id'],f));print('OVERLAP_SOURCE',i,'HOLD' if result['hold'] else 'PASS',flush=True)
 helper.check_inputs(identities)
 save=dict(status='screened_only_no_training_admission_or_model_measurement',inputs=identities,source_reviews_joint_accepted=accepted,protected_unique_photos=len(protected),protected_records=protected,permanent_monitor_photos=92,original_benchmark_photos=246,records=results,
   limitations='Heuristic byte/pixel/rotation-aware dHash<=8/SIFT350 mutual ratio.7 RANSAC4px>=6inliers; every flag held, including near aliases among candidates. Not proof of independent source captures or absence from pretraining.')
 (RUN/'overlap-screen.json').write_text(json.dumps(save,indent=2)+'\n');print('OVERLAP_COMPLETE',sum(not r['hold'] for r in results),len(results),flush=True)
if __name__=='__main__':main()
