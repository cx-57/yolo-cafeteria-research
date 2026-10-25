"""Screen reviewed clean/litter photos against all protected sources and each other."""
from pathlib import Path
import hashlib
import importlib.util
import json

RUN=Path(__file__).resolve().parent;LITTER=RUN.parent/'taco-expansion-staging-v38'
BASE=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-capture-validation')
HELPER=RUN.parent/'data-followup-v35/screen_followup.py'
spec=importlib.util.spec_from_file_location('v37_overlap_helpers',HELPER);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)

def main():
 identities={}
 def bound(path):
  value,digest=helper.snapshot(path);identities[str(path)]=digest;return value
 for path in [HELPER,Path(__file__)]:identities[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
 frozen=bound(RUN.parent/'validation-review/frozen-files-verification.json')
 manifest=bound(BASE/'manifest.json');assert identities[str(BASE/'manifest.json')]==frozen['manifest_sha']
 expected_by_name={r['name']:r['sha'] for r in manifest['records']}
 for path in sorted(RUN.parent.glob('*/dataset.json')):
  for row in bound(path).get('added',[]):
   if row.get('image') and row.get('image_sha'):
    name=Path(row['image']).name;digest=row['image_sha']
    if name in expected_by_name:assert expected_by_name[name]==digest
    expected_by_name[name]=digest
 monitor=bound(RUN.parent/'monitor-campaign-v34/dataset.json')
 active=bound(RUN.parent/'reviewed-neck-monitor-v36/dataset.json')
 assert monitor['monitor_records']==active['monitor_records'] and len(active['monitor_records'])==36
 targets=[(('benchmark/' if r['split']=='val' else 'historical-training/')+r['name'],BASE/'images'/r['split']/r['name'],r['sha']) for r in manifest['records']]
 targets.extend(('permanent-monitor/'+r['id'],Path(r['image']),r['image_sha']) for r in active['monitor_records'])
 listing=RUN.parent/'reviewed-neck-monitor-v36/training-exposures.txt';raw=listing.read_bytes();identities[str(listing)]=hashlib.sha256(raw).hexdigest();assert identities[str(listing)]==active['training_listing_sha']
 targets.extend(('current-training/'+Path(p).name,Path(p),expected_by_name[Path(p).name]) for p in sorted(set(raw.decode().splitlines())))
 candidates=[]
 for folder,kind in [(RUN,'openimages'),(LITTER,'taco-unofficial')]:
  rows=bound(folder/'downloads.json');first=bound(folder/'first-review.json');second=bound(folder/'second-review.json')
  assert second['first_review_sha256']==identities[str(folder/'first-review.json')]
  for index in second['accepted_indices']:
   row=rows[index];review=first['records'][index]
   assert review['index']==index and review['id']==row['id'] and review['sha']==row['sha'] and review['decision']=='candidate'
   candidates.append(dict(row,source_kind=kind,original_index=index))
 prepared=[];seen=set();protected=[]
 for name,path,expected in targets:
  digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==expected,'Protected photo changed: '+str(path)
  identities[str(path)]=digest;protected.append(dict(name=name,image=str(path),sha=digest))
  if digest not in seen:prepared.append((name,helper.protected_features(path,expected)));seen.add(digest)
 print('VERIFIED_PROTECTED_PHOTOS',len(seen),flush=True);results=[]
 for row in candidates:
  f=helper.features(row['image']);assert f['sha']==row['sha'];identities[row['image']]=f['sha']
  result=dict(id=row['id'],source_kind=row['source_kind'],original_index=row['original_index'],image=row['image'],sha=f['sha'],pixels=f['pixels'],**helper.compare(f,prepared))
  results.append(result);prepared.append(('candidate/'+row['id'],f));print('SOURCE_OVERLAP',row['source_kind'],row['original_index'],row['id'],'HOLD' if result['hold'] else 'PASS',flush=True)
 helper.check_inputs(identities)
 output=dict(status='screened_only_no_admission_or_model_measurements',inputs=identities,protected_records=protected,protected_unique_photos=len(seen),benchmark_images=246,permanent_monitor_images=36,records=results,
             limitations='All byte/pixel/dHash<=8/SIFT350 mutual ratio0.7 RANSAC4px>=6inlier flags held. Source/capture independence remains heuristic. Photos are not admitted, split or measured by model scores.')
 (RUN/'combined-overlap-screen.json').write_text(json.dumps(output,indent=2)+'\n')
 print('SOURCE_SCREEN_COMPLETE',sum(not r['hold'] for r in results),len(results),flush=True)

if __name__=='__main__':main()
