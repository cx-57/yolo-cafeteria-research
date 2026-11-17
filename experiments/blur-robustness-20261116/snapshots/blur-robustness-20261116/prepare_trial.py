import json,sys,shutil,hashlib
from pathlib import Path
import cv2,yaml
R=Path(__file__).resolve().parent;REPO=R.parents[2];A=REPO.parent/'yolo-cafeteria-research-assets';sys.path.insert(0,str(REPO/'scripts'))
from training import inspect_dataset
from specialists import sha
source=A/'artifacts/cumulative20-100views-20261031/train/resolved-data.yaml'
def main():
 info=inspect_dataset(source);out=A/'artifacts/blur-training-data-20261116';out.mkdir(exist_ok=False);counts={};files=[]
 for i,r in enumerate(info['records']['train']):
  im=cv2.imread(r['image']);assert im is not None;condition=['clean','gaussian','downsample'][int(r['sha256'][:8],16)%3]
  if condition=='gaussian':im=cv2.GaussianBlur(im,(9,9),2.)
  elif condition=='downsample':
   h,w=im.shape[:2];im=cv2.resize(cv2.resize(im,(max(1,w//3),max(1,h//3)),interpolation=cv2.INTER_AREA),(w,h),interpolation=cv2.INTER_LINEAR)
  name=f'{i:06d}';ip=out/'images/train'/(name+'.png');lp=out/'labels/train'/(name+'.txt');ip.parent.mkdir(parents=True,exist_ok=True);lp.parent.mkdir(parents=True,exist_ok=True);assert cv2.imwrite(str(ip),im);shutil.copy2(r['label'],lp)
  counts[condition]=counts.get(condition,0)+1;files.append(dict(source=r,image=str(ip),image_sha256=sha(ip),label=str(lp),label_sha256=sha(lp),condition=condition))
 data=dict(info['data'],train=str(out/'images/train'));(out/'data.yaml').write_text(yaml.safe_dump(data));(out/'receipt.json').write_text(json.dumps(dict(source_yaml_sha256=sha(source),records=files,counts=counts,scope='Training-only deterministic degradation; unchanged labels and source bytes; every source photo used once per epoch; validation/test membership unchanged.'),indent=2)+'\n')
 for name in ['control','candidate']:
  profile=yaml.safe_load((REPO/'configs/train-small-food-control.yaml').read_text());(R/f'{name}.yaml').write_text(yaml.safe_dump(profile))
 protocol=dict(parent=str(A/'artifacts/cumulative20-100views-20261031/train/weights/best.pt'),parent_sha256=sha(A/'artifacts/cumulative20-100views-20261031/train/weights/best.pt'),control_data=str(source),candidate_data=str(out/'data.yaml'),epochs=2,selected_checkpoint='last.pt at matched two-epoch cap',seed=42,confidence=.60,iou=.50,benchmark_sha256=sha(R/'benchmark/freeze.json'),acceptance='Blur-condition recall gain>=.02 absolute; precision loss<=.02. Clean precision and recall loss<=.02. Report every class and reject material class regressions. Synthetic development score alone does not certify phone-photo robustness.',user_screenshots_training=False,automatic_promotion=False)
 (R/'trial-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');print(counts)
if __name__=='__main__':main()
