import json,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent;A=R.parents[2].parent/'yolo-cafeteria-research-assets';last=0
while True:
 states={}
 for name in ['control','candidate']:
  p=A/f'artifacts/blur-neck-{name}-20261116/status.json'
  try:states[name]=json.loads(p.read_text()) if p.exists() else {}
  except json.JSONDecodeError:states[name]={}
 if any(s.get('state')=='failed' for s in states.values()):raise RuntimeError(states)
 if all(s.get('state')=='complete' for s in states.values()):break
 if time.monotonic()-last>30:print('WAIT',states,flush=True);last=time.monotonic()
 time.sleep(5)
print('Both matched trials complete; starting frozen comparison',flush=True)
with (R/'comparison.log').open('x') as log:subprocess.run([sys.executable,str(R/'compare_trials.py')],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=R.parents[2])
print('Comparison completed; inspect comparison.json',flush=True)
