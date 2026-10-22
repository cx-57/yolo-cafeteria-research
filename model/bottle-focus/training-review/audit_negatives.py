"""Audit training-only bottle absences; never rewrite original labels or validation."""
from pathlib import Path
import importlib.util,json
from PIL import Image,ImageDraw,ImageOps
from ultralytics import YOLO
RUN=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('bottle_common',RUN.parent/'plastic-specialist-s-v2/run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);s=m.s
weights=RUN.parent/'verified-incumbent/bottle.pt';proof=s.load(RUN.parent/'verified-incumbent/verification.json');assert s.sha(weights)==proof['weights_sha']
records=[r for r in m.MANIFEST['records'] if r['split']=='train' and not r['rows']]
assert len(records)==754
absence_flags={str(flag):sum(r['verified_absence']==flag for r in records) for flag in [False,True]}
lookup={r['name']:r for r in records}
listing=RUN/'negative-training-photos.txt';listing.write_text('\n'.join(str(m.DATA/'images/train'/r['name']) for r in records)+'\n')
found=[];seen=[]
model=YOLO(str(weights))
for i,result in enumerate(model.predict(source=str(listing),stream=True,device='mps',imgsz=640,batch=4,conf=.3,iou=.7,classes=[0],augment=False,verbose=False)):
    name=Path(result.path).name;assert name in lookup;seen.append(name)
    if len(result.boxes):
        boxes=[dict(xyxy=[float(x) for x in box],confidence=float(score)) for box,score in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist())]
        found.append(dict(name=name,image=str(m.DATA/'images/train'/name),sha=lookup[name]['sha'],boxes=boxes,max_confidence=max(x['confidence'] for x in boxes),status='requires_visual_review; predictions are not labels'))
    if (i+1)%100==0:print('TRAIN_NEGATIVE_AUDIT',i+1,'candidates',len(found),flush=True)
assert len(seen)==len(set(seen))==754 and set(seen)==set(lookup)
assert s.sha(weights)==proof['weights_sha']
s.validate_configuration(m.DATA,m.MANIFEST['records']);s.validate(m.MANIFEST['records'],m.DATA)
found.sort(key=lambda r:r['max_confidence'],reverse=True)
s.save(RUN/'negative-audit.json',dict(weights_sha=proof['weights_sha'],training_negative_images=754,original_verified_absence_flags=absence_flags,review_candidates=len(found),confidence=.3,nms_iou=.7,imgsz=640,test_time_augmentation=False,validation_used_for_selection=False,predictions_are_ground_truth=False,records=found))
for start in range(0,min(len(found),48),6):
    sheet=Image.new('RGB',(1536,1050),'white');d=ImageDraw.Draw(sheet)
    for j,r in enumerate(found[start:start+6]):
        im=Image.open(r['image']).convert('RGB');draw=ImageDraw.Draw(im)
        for b in r['boxes']:draw.rectangle(b['xyxy'],outline='red',width=max(2,im.width//220))
        im=ImageOps.contain(im,(504,470));x=j%3*512;y=j//3*525;sheet.paste(im,(x,y));d.text((x+2,y+475),f"{start+j}: {r['name'][-45:]} {r['max_confidence']:.2f}",fill='black')
    sheet.save(RUN/f'negative-review-{start:03d}.jpg',quality=90)
print('TRAIN_NEGATIVE_AUDIT_COMPLETE',754,'candidates',len(found),flush=True)
