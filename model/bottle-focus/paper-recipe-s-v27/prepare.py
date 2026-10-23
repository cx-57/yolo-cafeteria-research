"""Pin the already identity-checked original COCO-to-bottle small-model projection."""
from pathlib import Path
import hashlib,json,shutil
RUN=Path(__file__).resolve().parent
SOURCE=RUN.parent/'plastic-specialist-s-v2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
proof=json.loads((SOURCE/'initialization.json').read_text())
source=SOURCE/'initial.pt'
assert proof['COCO_class']==39 and proof['decoded_output_preserved']
assert sha(source)==proof['weights_sha']=='b28f055085711a1e264f5420aa90442fc57f6384535cbb0252c75ee38a155cb8'
assert sha(Path(proof['source']))==proof['source_sha']
initial=RUN/'initial.pt';assert not initial.exists()
shutil.copy2(source,initial);assert sha(initial)==proof['weights_sha']
result=dict(architecture='YOLO11s',source=str(source),source_sha=sha(source),weights_sha=sha(initial),COCO_projection=proof)
(RUN/'initialization.json').write_text(json.dumps(result,indent=2)+'\n')
print('PAPER_CONTROL_INITIALIZED',result['weights_sha'],flush=True)
