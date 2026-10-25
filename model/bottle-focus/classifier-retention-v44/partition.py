"""Whole-group exclusions and narrowly authorized training-only label correction."""
KEYS=['id','image_sha','pixels','source_group']
def keep(records,monitors,held_groups,screen_held_ids):
 forbidden={key:{r[key] for r in monitors} for key in KEYS}
 result=[r for r in records if r['source_group'] not in held_groups and r['id'] not in screen_held_ids and not any(r[k] in forbidden[k] for k in KEYS)]
 assert len({r['id'] for r in result})==len(result)
 return result
def reviewed_rows(record,first,second):
 assert second['full_scene_reviewed'] is True and second['human_verified'] is False
 assert first['index'] in second['accepted_indices'] and first['image_sha']==record['image_sha'] and first['rows']==record['rows']
 if first['decision']=='correct_to_negative':
  assert first['index'] in second['negative_indices'] and first['index'] in second['training_only_correction_indices'] and first['verified_absence'] is True and first['approved_rows']==[]
  return []
 assert first['decision']=='retain' and first['approved_rows']==record['rows']
 return record['rows']
