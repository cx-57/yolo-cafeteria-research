"""Exercise fresh initialization, reuse, and rejection of foreign weights."""
from pathlib import Path
import importlib.util,json,tempfile
RUN=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('feature_trial',RUN/'run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
expected='edab60c58fb8a1e941557b4bd84955a5208b75b476e0a587854d3e679e7d1c63'
with tempfile.TemporaryDirectory(prefix='bottle-initialization-') as temporary:
    m.RUN=Path(temporary);m.INITIAL=m.RUN/'initial.pt'
    m.initialize()
    assert m.s.sha(m.INITIAL)==expected
    ledger=m.s.load(m.RUN/'initialization.json');assert ledger['weights_sha']==expected
    assert Path(ledger['source']).resolve()==RUN.parent/'gentle-feature-adaptation-v6/phase8-best_bottle.pt'
    before=(m.RUN/'initialization.json').read_bytes();m.initialize();assert (m.RUN/'initialization.json').read_bytes()==before
    m.INITIAL.unlink();m.INITIAL.write_bytes(b'foreign checkpoint')
    try:m.initialize()
    except AssertionError:pass
    else:raise AssertionError('Foreign existing initialization was accepted')
assert m.s.sha(RUN.parent/'gentle-feature-adaptation-v6/phase8-best_bottle.pt')==expected
m.s.save(RUN/'initialization-probe.json',dict(fresh_copy=True,reuse_same_provenance=True,foreign_checkpoint_rejected=True,source_unchanged=True,weights_sha=expected))
print('INITIALIZATION_PROBE_PASSED',flush=True)
