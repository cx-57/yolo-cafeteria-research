"""One predeclared 50% weight-interpolation follow-up; no test selection."""
import json,sys,shutil
from pathlib import Path
R=Path(__file__).resolve().parent;REPO=R.parents[2];A=REPO.parent/'yolo-cafeteria-research-assets';sys.path.insert(0,str(REPO/'scripts'))
from specialists import sha

def main():
 import torch
 from ultralytics import YOLO
 torch.set_num_threads(2)
 original=json.loads((R/'comparison.json').read_text());assert not original['accept']
 out=R.parent/'blur-interpolation-20261116';out.mkdir(exist_ok=False);(out/'weights').mkdir()
 protocol=json.loads((R/'trial-protocol.json').read_text())
 hypothesis={'alpha':.5,'rationale':'Blur candidate gains recall but loses precision. Fixed equal interpolation toward unchanged parent may retain some blur adaptation while limiting drift. Apply same blend to matched clean control; no alpha sweep.','development_origin_sha256':sha(R/'comparison.json'),'confirmation_opened':False,'training':False,'parent':protocol['parent'],'source_proof':original['proof'],'gates':'Identical original aggregate/AP gates plus additive per-class operating retention; no relaxation.','automatic_promotion':False}
 (out/'protocol.json').write_text(json.dumps(hypothesis,indent=2)+'\n')
 parent=YOLO(protocol['parent']).model.state_dict();proof={}
 for name in ['control','candidate']:
  source=original['proof'][name];assert sha(source['checkpoint'])==source['sha256'];model=YOLO(source['checkpoint']);state=model.model.state_dict();merged={}
  assert state.keys()==parent.keys()
  for key,value in state.items():
   merged[key]=(.5*value.float()+.5*parent[key].float()).to(value.dtype) if value.is_floating_point() else parent[key].clone()
  model.model.load_state_dict(merged,strict=True);p=out/'weights'/f'{name}.pt';model.save(str(p));proof[name]={'source_checkpoint_sha256':source['sha256'],'checkpoint':str(p),'sha256':sha(p)}
 (out/'interpolation-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
 for file in ['trial-protocol.json','inference-config.json','class-retention-amendment.json','retention.py']:
  shutil.copy2(R/file,out/file)
 (out/'benchmark').symlink_to(R/'benchmark',target_is_directory=True)
 script=(R/'compare_trials.py').read_text().replace("checkpoint=folder/'weights/last.pt';recipe_digest=sha(folder/'recipe.json')","checkpoint=R/'weights'/f'{name}.pt';recipe_digest=sha(folder/'recipe.json')")
 script=script.replace("report=dict(inference_config_sha256", "report=dict(interpolation_protocol_sha256=sha(R/'protocol.json'),interpolation_proof_sha256=sha(R/'interpolation-proof.json'),inference_config_sha256")
 (out/'compare_trials.py').write_text(script)
 print('Prepared fixed halfway interpolation',proof,flush=True)
if __name__=='__main__':main()
