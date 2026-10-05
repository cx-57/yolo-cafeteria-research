from pathlib import Path
from src.inference.disposal_engine import DisposalEngine

def test_tbd_rules_abstain():
    engine = DisposalEngine(Path("configs/disposal_rules.yaml"))
    assert engine.classify("food") == "manual_check"

def test_unknown_class_abstains():
    engine = DisposalEngine(Path("configs/disposal_rules.yaml"))
    assert engine.classify("not_a_real_class") == "manual_check"
