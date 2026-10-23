"""Pin the immutable phase-six YOLO11m head checkpoint for feature adaptation."""
from pathlib import Path
import hashlib,json,shutil
RUN=Path(__file__).resolve().parent
SOURCE=RUN.parent/'medium-head-v24'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
proof=json.loads((SOURCE/'phase6-selected.json').read_text())
original=json.loads((SOURCE/'initialization.json').read_text())
source=SOURCE/'phase6-best_bottle.pt'
assert original['architecture']=='YOLO11m'
assert sha(source)==proof['weights_sha']
assert proof['validation_images']==246 and proof['bottle_objects']==67
assert proof['source_dataset_fingerprint']=='6668d2b7f0b16243640c14efdbb6bdffbb7f83fc880459befd58e8afcbfa2296'
assert proof['validation_protocol']=='29423591896198d41f5bd16392b51f06c7c3d7254f8a903b3dda74569cfe7de7'
initial=RUN/'initial.pt';assert not initial.exists()
shutil.copy2(source,initial);assert sha(initial)==proof['weights_sha']
result=dict(architecture='YOLO11m',source=str(source),source_sha=sha(source),weights_sha=sha(initial),source_verification=proof,pretraining=original)
(RUN/'initialization.json').write_text(json.dumps(result,indent=2)+'\n')
print('FEATURE_INITIALIZATION_PINNED',result['weights_sha'],flush=True)
