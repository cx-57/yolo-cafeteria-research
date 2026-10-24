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

proposal=json.loads((RUN/'monitor-source-proposal.json').read_text());records=proposal['records']
manifest=json.loads((DATA/'manifest.json').read_text())
selected_names={Path(r['source_image']).name for r in records if r['role']=='positive'}
current={Path(p).name for p in (RUN.parent/'reviewed-background-v33/training-exposures.txt').read_text().splitlines()}
training=[str(DATA/'images/train'/r['name']) for r in manifest['records'] if r['split']=='train' and r['name'] not in current]
targets=[('training/'+Path(p).name,Path(p)) for p in training]
prepared=[]
for name,path in targets:prepared.append((name,hashlib.sha256(path.read_bytes()).hexdigest(),features(path)))
results=[]
for r in records:
    assert hashlib.sha256(Path(r['source_image']).read_bytes()).hexdigest()==r['image_sha']
    pixel,dh,kp,desc=features(r['source_image']);assert pixel==r['pixels'];matches=[];dhash_matches=[];exact=[];nearest=(65,None)
    for name,digest,(other_pixel,other_dh,other_kp,other_desc) in prepared:
        if digest==r['image_sha'] or pixel==other_pixel:exact.append(name)
        distance=min((a^b).bit_count() for a in dh for b in other_dh)
        if distance<nearest[0]:nearest=(distance,name)
        if distance<=8:dhash_matches.append(dict(name=name,distance=distance))
        if desc is None or other_desc is None:continue
        forward={a.queryIdx:a.trainIdx for pair in matcher.knnMatch(desc,other_desc,k=2) if len(pair)==2 for a,b in [pair] if a.distance<.7*b.distance}
        reverse={a.queryIdx:a.trainIdx for pair in matcher.knnMatch(other_desc,desc,k=2) if len(pair)==2 for a,b in [pair] if a.distance<.7*b.distance}
        mutual=[(a,b) for a,b in forward.items() if reverse.get(b)==a]
        if len(mutual)<6:continue
        a=np.float32([kp[x].pt for x,y in mutual]);b=np.float32([other_kp[y].pt for x,y in mutual]);_,mask=cv2.findHomography(a,b,cv2.RANSAC,4.);inliers=int(mask.sum()) if mask is not None else 0
        if inliers>=6:matches.append(dict(name=name,mutual=len(mutual),inliers=inliers))
    implicated=set(exact)|{x['name'] for x in dhash_matches+matches}
    result=dict(id=r['id'],image_sha=r['image_sha'],pixels=pixel,nearest_dhash=nearest,exact_matches=exact,dhash_matches=dhash_matches,feature_matches=matches,benchmark_hold=any(x.startswith('benchmark/') for x in implicated),related_training_photos=sorted(x.split('/',1)[1] for x in implicated if x.startswith('training/')))
    results.append(result);print('MONITOR_SOURCE_SCREEN',r['id'],'BENCHMARK_HOLD',result['benchmark_hold'],'TRAINING_HOLDS',len(result['related_training_photos']),flush=True)
(RUN/'historical-overlap-screen.json').write_text(json.dumps(dict(proposal_sha256=hashlib.sha256((RUN/'monitor-source-proposal.json').read_bytes()).hexdigest(),benchmark_images=0,training_photos_screened=len(training),records=results,limitations='Byte/pixel, rotation-aware dHash<=8 and mutual SIFT350/RANSAC>=6inliers are conservative capture heuristics, not proof of full source independence. Related training groups will be excluded; benchmark overlap candidates are held from monitor.'),indent=2)+'\n')
