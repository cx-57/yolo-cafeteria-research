"""Balance reviewed original context photos more strongly than external additions."""
from pathlib import Path
from collections import Counter
import importlib.util,yaml
RUN=Path(__file__).resolve().parent
BASE=RUN.parent/'scope-clean-feature-v14'
spec=importlib.util.spec_from_file_location('scope_dataset',BASE/'build.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
m=base.m;s=base.s

def expected():
    source=base.verify()
    old=Counter((RUN.parent/'context-resampling-v5/training-exposures.txt').read_text().splitlines())
    context={Path(p).name for p,n in old.items() if n==4};assert len(context)==124
    unique=sorted(set((BASE/'training-exposures.txt').read_text().splitlines()))
    assert sum(Path(p).name in context for p in unique)==114
    listing=[p for p in unique for _ in range(8 if Path(p).name in context else 1)]
    assert len(unique)==1098 and len(listing)==1896
    return source,listing

def verify():
    source,listing=expected();audit=s.load(RUN/'dataset.json')
    assert s.sha(BASE/'dataset.json')==audit['base_dataset_audit_sha']
    assert (RUN/'training-exposures.txt').read_text().splitlines()==listing
    assert s.sha(RUN/'training-exposures.txt')==audit['training_listing_sha']
    assert yaml.safe_load((RUN/'data.yaml').read_text())==audit['config']
    assert audit['validation_images']==246 and audit['validation_bottle_boxes']==67
    return audit

if __name__=='__main__':
    source,listing=expected();p=RUN/'training-exposures.txt';p.write_text('\n'.join(listing)+'\n')
    config=dict(path=str(m.DATA),train=str(p),val=str(m.DATA/'images/val'),names=['bottle'])
    (RUN/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    audit=dict(source,base_dataset_audit_sha=s.sha(BASE/'dataset.json'),training_unique_images=len(set(listing)),training_exposures=len(listing),training_listing_sha=s.sha(p),config=config,context_positive_photos=114,context_positive_repeats=8,external_positive_repeats=1)
    s.save(RUN/'dataset.json',audit);verify();print('CONTEXT_BALANCE_VERIFIED',len(set(listing)),len(listing),'context',114,'external',57,flush=True)
