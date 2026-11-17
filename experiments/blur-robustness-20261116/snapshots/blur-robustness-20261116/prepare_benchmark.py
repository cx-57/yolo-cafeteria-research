import json,sys,hashlib,shutil
from pathlib import Path
import cv2,yaml
R=Path(__file__).resolve().parent;REPO=R.parents[2];A=REPO.parent/'yolo-cafeteria-research-assets';sys.path.insert(0,str(REPO/'scripts'))
from training import inspect_dataset
from specialists import sha
SOURCE=A/'artifacts/cumulative20-100views-20261031/train/resolved-data.yaml'
def main():
 info=inspect_dataset(SOURCE);out=R/'benchmark';out.mkdir(exist_ok=False);records=[]
 for i,r in enumerate(info['records']['val']):
  im=cv2.imread(r['image']);assert im is not None
  for condition in ['clean','gaussian','downsample']:
   image=im.copy()
   if condition=='gaussian':image=cv2.GaussianBlur(image,(9,9),2.)
   elif condition=='downsample':
    h,w=image.shape[:2];image=cv2.resize(cv2.resize(image,(max(1,w//3),max(1,h//3)),interpolation=cv2.INTER_AREA),(w,h),interpolation=cv2.INTER_LINEAR)
   name=f'{condition}-{i:04d}';p=out/'images'/(name+'.png');p.parent.mkdir(exist_ok=True);assert cv2.imwrite(str(p),image)
   records.append(dict(name=name,image_path=str(p),sha256=sha(p),source_image=r['image'],source_sha256=r['sha256'],group=r['sha256'],condition=condition,rows=Path(r['label']).read_text().splitlines(),known_classes=list(range(7))))
 parent=A/'artifacts/cumulative20-100views-20261031/train/weights/best.pt'
 manifest=dict(records=records,parent=str(parent),parent_sha256=sha(parent),source_yaml_sha256=sha(SOURCE),confidence=.60,minimum_iou=.50,training_images=len(info['records']['train']),validation_source_images=len(info['records']['val']),scope='Fixed synthetic blur development benchmark; no user screenshot in training or benchmark; no real-camera accuracy certification.',degradations=dict(gaussian=dict(kernel=9,sigma=2),downsample=dict(factor=3,down='area',up='linear')))
 (out/'freeze.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Frozen',len(records),'views from',len(info['records']['val']),'source photos')
if __name__=='__main__':main()
