"""Reuse the tested frozen-benchmark joint-metric policy."""
from pathlib import Path
import importlib.util
spec=importlib.util.spec_from_file_location('shared_joint_policy',Path(__file__).resolve().parents[1]/'joint-metric-v30/policy.py')
shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared)
metrics=shared.metrics
target_met=shared.target_met
save_roles=shared.save_roles
frontier=shared.frontier
verified_records=shared.verified_records
phase_outcome=shared.phase_outcome
