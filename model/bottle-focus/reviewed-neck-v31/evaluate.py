"""Reuse immutable, fixed-configuration joint-metric verification."""
from pathlib import Path
import importlib.util
spec=importlib.util.spec_from_file_location('shared_joint_evaluation',Path(__file__).resolve().parents[1]/'joint-metric-v30/evaluate.py')
shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared)
evaluate=shared.evaluate
