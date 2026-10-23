"""Hold material/scope-ambiguous training photos without altering original data."""
from pathlib import Path
import importlib.util,yaml
RUN=Path(__file__).resolve().parent
BASE=RUN.parent/'expanded-context-v13'
REVIEW=RUN.parent/'training-review/context-full/reviews.json'
spec=importlib.util.spec_from_file_location('expanded_dataset',BASE/'build.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
m=base.m;s=base.s

def expected():
    source=base.verify();review=s.load(REVIEW);originals={r['name']:r for r in m.MANIFEST['records']}
    holds=[r for r in review['records'] if r['quality_hold']]
    for r in review['records']:
        original=originals[r['name']]
        assert original['split']=='train' and original['sha']==r['image_sha'] and original['rows']==r['rows']
    assert len(review['records'])==124 and len(holds)==10 and sum(len(r['rows']) for r in holds)==14
    listing=[p for p in (BASE/'training-exposures.txt').read_text().splitlines() if Path(p).name not in {r['name'] for r in holds}]
    assert len(set(listing))==1098 and len(listing)==1611
    return source,holds,listing

def verify():
    source,holds,listing=expected();audit=s.load(RUN/'dataset.json')
    assert s.sha(BASE/'dataset.json')==audit['base_dataset_audit_sha'] and s.sha(REVIEW)==audit['scope_review_sha']
    assert (RUN/'training-exposures.txt').read_text().splitlines()==listing
    assert s.sha(RUN/'training-exposures.txt')==audit['training_listing_sha']
    assert yaml.safe_load((RUN/'data.yaml').read_text())==audit['config']
    assert audit['validation_images']==246 and audit['validation_bottle_boxes']==67
    return audit

if __name__=='__main__':
    source,holds,listing=expected();p=RUN/'training-exposures.txt';p.write_text('\n'.join(listing)+'\n')
    config=dict(path=str(m.DATA),train=str(p),val=str(m.DATA/'images/val'),names=['bottle'])
    (RUN/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    audit=dict(source,base_dataset_audit_sha=s.sha(BASE/'dataset.json'),scope_review_sha=s.sha(REVIEW),training_unique_images=len(set(listing)),training_exposures=len(listing),training_listing_sha=s.sha(p),config=config,training_quality_holds=source['training_quality_holds']+holds,held_original_bottle_boxes=14,training_unique_bottle_boxes=451)
    s.save(RUN/'dataset.json',audit);verify();print('SCOPE_DATA_VERIFIED',len(set(listing)),len(listing),'quality holds',15,'bottle boxes',451,flush=True)
