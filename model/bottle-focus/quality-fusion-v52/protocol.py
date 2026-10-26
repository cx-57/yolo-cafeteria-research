"""One inference setting, four-cohort gates and settings-bound attempt lock."""
from pathlib import Path
import json
import math

def valid(values):
 assert len(values)==8 and all(type(v) in (int,float) and math.isfinite(v) for v in values)
 assert all(0<=values[i+1]<=values[i]<=1 for i in [0,2,4,6])
def eligible(candidate,baseline):
 valid(candidate);valid(baseline);d=[a-b for a,b in zip(candidate,baseline,strict=True)]
 return min(d[:2])>=-.005-1e-12 and min(d[2:])>=0 and sum(d[2:])/6>=.01-1e-12
def bind_attempt(path,identity):
 path=Path(path)
 if path.exists():assert json.loads(path.read_text())==identity,'Attempt settings changed';return
 temp=path.with_suffix('.partial');temp.write_text(json.dumps(identity,indent=2)+'\n');temp.replace(path)
def receipt(report,expected):
 assert all(report.get(k)==v for k,v in expected.items()),'Saved score identity mismatch'
 a,b=report['AP50'],report['AP50_95'];assert all(type(v) in (int,float) and math.isfinite(v) for v in [a,b]) and 0<=b<=a<=1
 return report
