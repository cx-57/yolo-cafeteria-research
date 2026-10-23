"""Exploratory replay of saved predictions; never a fresh-model verification."""
from pathlib import Path
from collections import defaultdict
from types import SimpleNamespace
import json,sys,hashlib
import numpy as np
import torch
from PIL import Image
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import DetMetrics,box_iou
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'rotation-fusion-v17'))
from vendor_wbf import weighted_boxes_fusion
DATA=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-capture-validation')
files=[ROOT/'context-head-soup-v23/evaluations/ordinary-batch1-iou0.5-mps/predictions.json',ROOT/'medium-head-v24/evaluations/selected/predictions.json']
groups=[]
for f in files:
    group=defaultdict(list)
    for p in json.loads(f.read_text()):
        assert p['category_id']==1  # Native export maps model class 0 to COCO ID 1.
        group[p['file_name']].append(p)
    groups.append(group)
matcher=SimpleNamespace(iouv=torch.linspace(.5,.95,10))
metrics=[DetMetrics(names={0:'bottle'}),DetMetrics(names={0:'bottle'})]
count=0
for image in sorted((DATA/'images/val').iterdir()):
    if image.suffix.lower() not in {'.jpg','.jpeg','.png'}:continue
    count+=1
    with Image.open(image) as im:width,height=im.size
    boxes=[];scores=[]
    for group in groups:
        rows=group[image.name]
        b=np.array([[r['bbox'][0]/width,r['bbox'][1]/height,(r['bbox'][0]+r['bbox'][2])/width,(r['bbox'][1]+r['bbox'][3])/height] for r in rows],dtype=np.float32).reshape(-1,4).clip(0,1)
        boxes.append(b);scores.append(np.array([r['score'] for r in rows],dtype=np.float32))
    fused,confidence,_=weighted_boxes_fusion(boxes,scores,[np.zeros(len(b)) for b in boxes],weights=[2,1],iou_thr=.55,skip_box_thr=.001,conf_type='avg',allows_overflow=False)
    gt=[]
    for row in (DATA/'labels/val'/(image.stem+'.txt')).read_text().splitlines():
        c,x,y,w,h=map(float,row.split());assert c==0;gt.append([x-w/2,y-h/2,x+w/2,y+h/2])
    target=torch.tensor(gt,dtype=torch.float32).reshape(-1,4)
    for metric,pred,conf in zip(metrics,[boxes[0],fused[:300]],[scores[0],confidence[:300]]):
        pred=torch.as_tensor(pred,dtype=torch.float32);classes=torch.zeros(len(pred));truth=torch.zeros(len(target))
        tp=BaseValidator.match_predictions(matcher,classes,truth,box_iou(target,pred))
        metric.update_stats(dict(tp=tp.numpy(),conf=np.array(conf),pred_cls=classes.numpy(),target_cls=truth.numpy(),target_img=np.unique(truth.numpy()),im_name=image.name))
assert count==246
results=[]
for metric in metrics:
    metric.process();assert metric.nt_per_class.tolist()==[67]
    results.append(dict(AP50=float(metric.box.ap50[0]),AP50_95=float(metric.box.map)))
expected=json.loads((ROOT/'context-head-soup-v23/ordinary-batch1-iou0.5-mps.json').read_text())['AP50']
report=dict(exploratory_replay_only=True,fresh_model_inference=False,validation_images=count,bottle_objects=67,small_replayed=results[0],ensemble_replayed=results[1],small_saved_model_AP50=expected,control_difference=results[0]['AP50']-expected,wbf=dict(weights=[2,1],iou_thr=.55,skip_box_thr=.001),sources=[dict(path=str(f),sha=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files],limitations='Rounded exported boxes/confidences and different source batch sizes/NMS settings. Any apparent gain requires live two-model validation on unchanged data; never promote from this screen.')
(ROOT/'training-review/model-fusion-screen.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
