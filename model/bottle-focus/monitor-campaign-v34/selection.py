"""Predeclared monitor-only choice and one distinct final benchmark candidate."""
from pathlib import Path
import json
import math

MAX_EPOCHS=9

def budget_finished(epoch):
 assert type(epoch) is int and 0<=epoch<=MAX_EPOCHS
 return epoch==MAX_EPOCHS

def choose(reports,monitor_fingerprint):
 assert reports and any(r['role']=='initial' for r in reports)
 for r in reports:
  assert r['metric_source']=='continuation_monitor' and r['monitor_fingerprint']==monitor_fingerprint
  a,b=r['AP50'],r['AP50_95']
  if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (a,b)) or not 0<=b<=a<=1:raise ValueError('Invalid monitor metrics')
  assert type(r['selected_epoch']) is int and 0<=r['selected_epoch']<=MAX_EPOCHS
 return max(reports,key=lambda r:(r['AP50']+r['AP50_95'],r['role']=='initial',-r['selected_epoch']))

def bind_benchmark(path,weights_sha,training_audit_sha):
 path=Path(path);expected=dict(weights_sha=weights_sha,training_audit_sha=training_audit_sha,distinct_candidate_limit=1)
 if path.exists():
  assert json.loads(path.read_text())==expected,'A different benchmark candidate/data was already bound'
  return False
 temporary=path.with_suffix('.json.partial');temporary.write_text(json.dumps(expected,indent=2)+'\n');temporary.replace(path)
 return True


def bind_campaign(path,declaration_sha):
 path=Path(path);record=dict(declaration_sha=declaration_sha)
 if path.exists():
  assert json.loads(path.read_text())==record,'Campaign declaration changed after monitor evaluation began'
  return False
 temporary=path.with_suffix('.json.partial');temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(path);return True
