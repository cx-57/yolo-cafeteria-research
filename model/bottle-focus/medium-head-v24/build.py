"""Reuse the exact verified context-balanced data for a different architecture."""
from pathlib import Path
import importlib.util
RUN=Path(__file__).resolve().parent
BASE=RUN.parent/'context-balance-v20'
spec=importlib.util.spec_from_file_location('context_dataset',BASE/'build.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
m=base.m;s=base.s

def verify():
    audit=base.verify()
    for name in ('dataset.json','data.yaml'):assert (RUN/name).read_bytes()==(BASE/name).read_bytes(),name
    return audit
