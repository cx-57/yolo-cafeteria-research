import json,sys
from pathlib import Path
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent/'generalization-20261116'))
from screen_and_freeze import features,near
p=R.parents[2].parent/'yolo-cafeteria-research-assets/artifacts/blur-robustness-control-20261116/recipe.json';records=json.loads(p.read_text())['dataset']['records']
fp={s:[dict(r,**features(r['image'])) for r in records[s]] for s in ['train','val']};matches=[]
for a in fp['val']:
 for b in fp['train']:
  if near(dict(a,sha256=a['sha256']),dict(b,sha256=b['sha256'])):matches.append(dict(validation=a['image'],training=b['image'],phash_distance=(a['phash']^b['phash']).bit_count(),dhash_distance=(a['dhash']^b['dhash']).bit_count()))
(R/'perceptual-audit.json').write_text(json.dumps(dict(matches=matches,train_images=len(fp['train']),val_images=len(fp['val']),policy='Aspect-independent exact/pixel or pHash<=4 and dHash<=6; source-name identity separately checked. Crops/session independence remain uncertain.'),indent=2)+'\n');print('Potential train/val matches',len(matches))
