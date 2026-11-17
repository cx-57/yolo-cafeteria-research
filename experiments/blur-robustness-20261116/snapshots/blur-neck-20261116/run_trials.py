import json,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent;REPO=R.parents[2];A=REPO.parent/'yolo-cafeteria-research-assets'
while not (R/'trial-protocol.json').exists():
 time.sleep(3)
p=json.loads((R/'trial-protocol.json').read_text())
for name in ['control','candidate']:
 output=A/f'artifacts/blur-neck-{name}-20261116';assert not output.exists()
 cmd=[sys.executable,str(REPO/'scripts/training.py'),'train','--data',p[name+'_data'],'--model',p['parent'],'--profile',str(R/f'{name}.yaml'),'--output',str(output),'--device','cpu']
 print('START',name,flush=True)
 with (R/f'train-{name}.log').open('x') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=REPO)
 print('COMPLETE',name,flush=True)
