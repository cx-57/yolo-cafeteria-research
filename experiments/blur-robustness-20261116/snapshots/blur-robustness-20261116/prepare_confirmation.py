"""Freeze source-disjoint confirmation photos before any candidate is selected."""
import json,re,sys
from pathlib import Path
import cv2
R=Path(__file__).resolve().parent;REPO=R.parents[2];A=REPO.parent/'yolo-cafeteria-research-assets';sys.path.insert(0,str(REPO/'scripts'));sys.path.insert(0,str(R.parent/'generalization-20261116'))
from specialists import sha
from screen_and_freeze import features,near

def key(p):
 n=Path(p).name
 while re.match(r'^\d{6}-',n):n=n[7:]
 n=n.split('.rf.')[0]
 return re.sub(r'__(train|test|val|valid|validation)__','__',n)
def main():
 out=R/'confirmation';out.mkdir(exist_ok=False)
 source=json.loads((A/'artifacts/blur-robustness-control-20261116/recipe.json').read_text())['dataset']['records'];known=[dict(r,**features(r['image'])) for split in ['train','val'] for r in source[split]];keys={key(r['image']) for r in known};selected=[];excluded=[]
 for r in source['test']:
  assert sha(r['image'])==r['sha256'] and sha(r['label'])==r['label_sha256'];r=dict(r,**features(r['image']));matches=[x['image'] for x in known+selected if near(r,x)]
  if key(r['image']) in keys or matches:excluded.append(dict(image=r['image'],matches=matches,reason='Normalized source identity or conservative perceptual/pixel/byte overlap with train/development/already selected test source'));continue
  selected.append(r);keys.add(key(r['image']))
 records=[]
 for i,r in enumerate(selected):
  im=cv2.imread(r['image']);assert im is not None
  for condition in ['clean','gaussian','downsample']:
   image=im.copy()
   if condition=='gaussian':image=cv2.GaussianBlur(image,(9,9),2.)
   elif condition=='downsample':
    h,w=image.shape[:2];image=cv2.resize(cv2.resize(image,(max(1,w//3),max(1,h//3)),interpolation=cv2.INTER_AREA),(w,h),interpolation=cv2.INTER_LINEAR)
   name=f'confirmation-{condition}-{i:04d}';p=out/'images'/(name+'.png');p.parent.mkdir(exist_ok=True);assert cv2.imwrite(str(p),image)
   records.append(dict(name=name,image_path=str(p),sha256=sha(p),source_image=r['image'],source_sha256=r['sha256'],source_label_sha256=r['label_sha256'],group=key(r['image']),condition=condition,rows=Path(r['label']).read_text().splitlines(),known_classes=list(range(7))))
 freeze=dict(records=records,source_images=len(selected),original_test_images=len(source['test']),excluded=excluded,confidence=.60,minimum_iou=.50,selection='Source identity and duplicate screening only, before any test model inference or development winner selection.',scope='One-shot confirmation using distinct source photos in previously unused test partition. No model/epoch/threshold selection on these results. Not a certified independent camera-session or historical-pretraining benchmark.',gate='Open only after a validation-selected configuration passes all predeclared development gates. Require Gaussian and downsample recall improvements>=.02 vs parent and precision loss<=.02; clean precision/recall loss<=.02; report allclassoutcomes and reject materialcleanclassregressions. Testfailurecannotbeusedtotunethe sameconfirmationcohort.')
 (out/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n');print('Frozen confirmation sources',len(selected),'views',len(records),'excluded',len(excluded),flush=True)
if __name__=='__main__':main()
