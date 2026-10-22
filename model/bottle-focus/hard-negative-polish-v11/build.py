"""Create a training-only quality hold and hard-negative exposure list."""
from pathlib import Path
from collections import Counter
import importlib.util,shutil
import yaml
RUN=Path(__file__).resolve().parent
BASE=RUN.parent/'taco-data-v10'
spec=importlib.util.spec_from_file_location('external_base',BASE/'build.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
m=base.m;s=base.s

def expected():
    source=base.verify();reviews=s.load(RUN/'negative-reviews.json')
    originals={r['name']:r for r in m.MANIFEST['records']}
    held=set();hard=[]
    for r in reviews['holds']+reviews['hard_negatives']:
        original=originals[r['name']]
        assert original['split']=='train' and not original['rows'] and original['sha']==r['image_sha']
    held={r['name'] for r in reviews['holds']}
    listing=[p for p in (BASE/'training-exposures.txt').read_text().splitlines() if Path(p).name not in held]
    for r in reviews['hard_negatives']:
        image=str(m.DATA/'images/train'/r['name']);assert listing.count(image)==1
        listing.extend([image]*(r['repeat']-1));hard.append(image)
    assert len(held)==5 and len(hard)==12
    assert len(set(listing))==1092 and len(listing)==1623
    assert not held & {Path(p).name for p in listing}
    return source,reviews,listing

def verify():
    source,reviews,listing=expected();audit=s.load(RUN/'dataset.json')
    assert s.sha(BASE/'dataset.json')==audit['base_dataset_audit_sha']
    assert s.sha(RUN/'negative-reviews.json')==audit['negative_review_sha']
    assert (RUN/'training-exposures.txt').read_text().splitlines()==listing
    assert s.sha(RUN/'training-exposures.txt')==audit['training_listing_sha']
    assert yaml.safe_load((RUN/'data.yaml').read_text())==audit['config']
    assert audit['validation_images']==246 and audit['validation_bottle_boxes']==67
    return audit

if __name__=='__main__':
    source,reviews,listing=expected()
    path=RUN/'training-exposures.txt';path.write_text('\n'.join(listing)+'\n')
    config=dict(path=str(m.DATA),train=str(path),val=str(m.DATA/'images/val'),names=['bottle'])
    (RUN/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    audit=dict(source,base_dataset_audit_sha=s.sha(BASE/'dataset.json'),negative_review_sha=s.sha(RUN/'negative-reviews.json'),training_unique_images=len(set(listing)),training_exposures=len(listing),training_listing_sha=s.sha(path),config=config,training_quality_holds=reviews['holds'],hard_negative_photos=len(reviews['hard_negatives']),hard_negative_repeats=4)
    s.save(RUN/'dataset.json',audit);verify()
    print('QUALITY_DATASET_VERIFIED',audit['training_unique_images'],audit['training_exposures'],'holds',5,'hard negatives',12,flush=True)
