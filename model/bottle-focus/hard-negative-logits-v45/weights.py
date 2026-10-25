"""Conservative source-family exclusions and reviewed-negative-only resampling."""
import re
def canonical_group(record):
 match=re.search(r'lvis_(\d+)',record['id'])
 return 'coco/'+str(int(match.group(1))) if match else record['source_group']
def reweight(records,monitors,held,approved_ids,extra):
 assert type(extra) is int and extra==8
 forbidden_groups={canonical_group(r) for r in monitors+held};forbidden={k:{r[k] for r in monitors+held} for k in ['id','image_sha','pixels']}
 result=[];used=set()
 for record in records:
  if canonical_group(record) in forbidden_groups or any(record[k] in forbidden[k] for k in forbidden):continue
  n=record['exposures']
  if record['id'] in approved_ids:
   assert record['rows']==[],'Only verified negative records may be upweighted'
   n+=extra;used.add(record['id'])
  result.append(dict(record,exposures=n))
 assert used==set(approved_ids),'Approved hard negative absent, held or reserved'
 return result

def validate_mining(records,candidates,parents):
 import math
 negative={r['id']:r for r in parents if not r['rows']}
 assert len(records)==len(negative) and {r['id'] for r in records}==set(negative)
 for r in records:
  original=negative[r['id']];assert all(r[k]==v for k,v in original.items()),'Stale or wrong source in mining'
  scores=[p['score'] for p in r['predictions']]
  assert all(math.isfinite(s) and .1<=s<=1 for s in scores)
  assert r['maximum_false_detection_score']==max(scores,default=0.)
 expected=sorted([r for r in records if r['predictions']],key=lambda r:(-r['maximum_false_detection_score'],r['id']))[:16]
 assert candidates==expected,'Candidate list differs from frozen teacher ranking'

def legacy_holds(records,first,second):
 assert first['human_verified'] is False and second['human_verified'] is False and first['full_scene_reviewed'] is True and second['full_scene_reviewed'] is True
 approved=second['accepted_indices'];assert len(approved)==len(set(approved)) and set(approved)<=set(range(len(first['records'])))
 lookup={r['id']:r for r in records};held=[]
 for index,review in enumerate(first['records']):
  record=lookup[review['id']];assert record['rows'] and review['index']==index and review['image_sha']==record['image_sha'] and review['rows']==record['rows']
  if index in approved:assert review['decision']=='retain'
  else:held.append(record)
 return held
