"""Ensure the local-feature screen detects a re-encoded validation photo."""
from pathlib import Path
import json
import cv2,numpy as np
from PIL import Image
from io import BytesIO
RUN=Path(__file__).resolve().parent
DATA=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-capture-validation')
manifest=json.loads((DATA/'manifest.json').read_text())
r=next(r for r in manifest['records'] if r['split']=='val' and r['rows'])
im=Image.open(DATA/'images/val'/r['name']).convert('RGB');im.thumbnail((640,640))
b=BytesIO();im.save(b,format='JPEG',quality=75);b.seek(0);reencoded=Image.open(b).convert('RGB')
sift=cv2.SIFT_create(nfeatures=350);matcher=cv2.BFMatcher();cv2.setNumThreads(2)
a,ad=sift.detectAndCompute(cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2GRAY),None)
b,bd=sift.detectAndCompute(cv2.cvtColor(np.asarray(reencoded),cv2.COLOR_RGB2GRAY),None)
f={x.queryIdx:x.trainIdx for pair in matcher.knnMatch(ad,bd,k=2) if len(pair)==2 for x,y in [pair] if x.distance<.7*y.distance}
r={x.queryIdx:x.trainIdx for pair in matcher.knnMatch(bd,ad,k=2) if len(pair)==2 for x,y in [pair] if x.distance<.7*y.distance}
mutual=[(x,y) for x,y in f.items() if r.get(y)==x]
_,mask=cv2.findHomography(np.float32([a[x].pt for x,y in mutual]),np.float32([b[y].pt for x,y in mutual]),cv2.RANSAC,4.)
assert int(mask.sum())>=6
proof=dict(control='JPEG re-encode of fixed validation image',mutual_matches=len(mutual),inliers=int(mask.sum()),threshold=6,passed=True,limitation='Detects this duplicate control; does not prove capture independence. Related birthday capture excluded by visual review.')
(RUN/'screen-positive-control.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
