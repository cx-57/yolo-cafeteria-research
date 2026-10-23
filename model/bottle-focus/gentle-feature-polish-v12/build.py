"""Reuse the verified v11 training copy without changing any examples."""
from pathlib import Path
import importlib.util
RUN=Path(__file__).resolve().parent
BASE=RUN.parent/'hard-negative-polish-v11'
spec=importlib.util.spec_from_file_location('quality_dataset',BASE/'build.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
m=base.m;s=base.s

def verify():
    audit=base.verify()
    for name in ('dataset.json','data.yaml'):
        assert (RUN/name).read_bytes()==(BASE/name).read_bytes(),name
    return audit

if __name__=='__main__':
    audit=verify();print('FEATURE_DATASET_VERIFIED',audit['training_unique_images'],audit['training_exposures'],flush=True)
