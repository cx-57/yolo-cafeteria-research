"""Reserve a fixed litter cohort from reviewed sources before model measurement."""
import random

SEED=20261025
SUPPORT_PER_ROLE=16

def split(positives,negatives,forbidden_ids=(),forbidden_shas=(),forbidden_pixels=(),forbidden_groups=()):
 if len(positives)<SUPPORT_PER_ROLE or len(negatives)<SUPPORT_PER_ROLE:
  raise ValueError('Need at least16 positive and16 negative photos before any measurement')
 assert all(r['rows'] for r in positives) and not any(r['rows'] for r in negatives)
 all_rows=positives+negatives
 for key,forbidden in [('id',forbidden_ids),('image_sha',forbidden_shas),('pixels',forbidden_pixels),('source_group',forbidden_groups)]:
  values=[r[key] for r in all_rows];assert all(values) and len(set(values))==len(values)
  assert not (set(values)&set(forbidden)),'Permanent monitor/source alias in candidate pool'
 generator=random.Random(SEED)
 selected=generator.sample(sorted(positives,key=lambda r:r['id']),SUPPORT_PER_ROLE)+generator.sample(sorted(negatives,key=lambda r:r['id']),SUPPORT_PER_ROLE)
 monitor=sorted(selected,key=lambda r:r['id']);ids={r['id'] for r in monitor}
 remaining=sorted([r for r in all_rows if r['id'] not in ids],key=lambda r:r['id'])
 for key in ['id','image_sha','pixels','source_group']:
  assert not ({r[key] for r in monitor}&{r[key] for r in remaining})
 return dict(seed=SEED,monitor=monitor,remaining=remaining,positive_photos=SUPPORT_PER_ROLE,negative_photos=SUPPORT_PER_ROLE,selected_using_model_scores=False)
