"""Use the exact verified V28 training/validation configuration without altering data."""
from pathlib import Path
import importlib.util
import yaml
RUN=Path(__file__).resolve().parent
BASE=RUN.parent/'reviewed-expansion-v28'
spec=importlib.util.spec_from_file_location('v29_base',BASE/'build.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
m=base.m;s=base.s
def verify():
    source=base.verify();audit=s.load(RUN/'dataset.json')
    assert audit['base_dataset_audit_sha']==s.sha(BASE/'dataset.json')
    assert audit['config']==source['config']==yaml.safe_load((RUN/'data.yaml').read_text())
    assert audit['training_listing_sha']==source['training_listing_sha']
    assert audit['training_unique_images']==1115 and audit['training_unique_bottle_boxes']==471 and audit['training_exposures']==1964
    assert audit['validation_images']==246 and audit['validation_bottle_boxes']==67
    return audit
if __name__=='__main__':
    source=base.verify()
    (RUN/'data.yaml').write_text(yaml.safe_dump(source['config'],sort_keys=False))
    s.save(RUN/'dataset.json',dict(source,base_dataset_audit_sha=s.sha(BASE/'dataset.json')))
    verify();print('V29_EXACT_V28_DATA_VERIFIED',1115,471,1964,flush=True)
