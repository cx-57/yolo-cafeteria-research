"""Screen fresh candidates against fixed validation using hashes and local features."""
from pathlib import Path
import hashlib,json
import cv2
import numpy as np
from PIL import Image,ImageOps
from audit import difference_hash
RUN=Path(__file__).resolve().parent
DATA=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-capture-validation')
cv2.setNumThreads(2)
sift=cv2.SIFT_create(nfeatures=350)
matcher=cv2.BFMatcher()
def features(path):
    with Image.open(path) as opened:
        im=ImageOps.exif_transpose(opened).convert('RGB')
        pixel=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
        dh=[difference_hash(im.rotate(k*90,expand=True)) for k in range(4)]
        im.thumbnail((640,640));gray=cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2GRAY)
        kp,desc=sift.detectAndCompute(gray,None)
        return pixel,dh,kp,desc
manifest=json.loads((DATA/'manifest.json').read_text())
known_sha={r['sha'] for r in manifest['records']}
known_pixels={r['pixels'] for r in manifest['records']}
validation=[]
for r in manifest['records']:
    if r['split']=='val':validation.append((r['name'],features(DATA/'images/val'/r['name'])))
results=[]
for r in json.loads((RUN/'lvis-round5-downloads.json').read_text()):
    if r['status']!='unreviewed':continue
    pixel,dh,kp,desc=features(r['image']);matches=[];nearest=(65,None)
    for name,(_,other_dh,other_kp,other_desc) in validation:
        distance=min((a^b).bit_count() for a in dh for b in other_dh)
        if distance<nearest[0]:nearest=(distance,name)
        if desc is None or other_desc is None:continue
        forward={a.queryIdx:a.trainIdx for pair in matcher.knnMatch(desc,other_desc,k=2) if len(pair)==2 for a,b in [pair] if a.distance<.7*b.distance}
        reverse={a.queryIdx:a.trainIdx for pair in matcher.knnMatch(other_desc,desc,k=2) if len(pair)==2 for a,b in [pair] if a.distance<.7*b.distance}
        mutual=[(a,b) for a,b in forward.items() if reverse.get(b)==a]
        if len(mutual)<6:continue
        a=np.float32([kp[x].pt for x,y in mutual]);b=np.float32([other_kp[y].pt for x,y in mutual])
        _,mask=cv2.findHomography(a,b,cv2.RANSAC,4.)
        inliers=int(mask.sum()) if mask is not None else 0
        if inliers>=6:matches.append(dict(name=name,mutual=len(mutual),inliers=inliers))
    held=r['sha'] in known_sha or pixel in known_pixels or nearest[0]<=8 or bool(matches)
    result=dict(id=r['id'],sha=r['sha'],pixels=pixel,exact_duplicate=r['sha'] in known_sha or pixel in known_pixels,nearest_validation_dhash=nearest,validation_feature_matches=matches,hold=held)
    results.append(result);print('SCREEN',r['id'],'HOLD' if held else 'PASS',nearest,matches,flush=True)
(RUN/'round5-duplicate-screen.json').write_text(json.dumps(dict(validation_images=len(validation),candidate_count=len(results),records=results,limitations='Heuristic capture/near-duplicate screen; not proof of complete source independence.'),indent=2)+'\n')
