from pathlib import Path
import json,hashlib,time
from ultralytics import YOLO
import utensil_packet as s
root=Path.cwd();run=root/'model/utensil-focus/utensil-90';weights=run/'utensil-yolo11x.pt';data=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/utensil-v12');m=s.load(data/'manifest.json')
s.validate(m['records'],data);s.validate_configuration(data,m['records'])
initial=s.load(run/'initialization.json');initial['architecture']=initial['architecture'].replace('YOLO11s','YOLO11x');s.save(run/'initialization.json',initial)
assert s.sha(weights)==initial['weights_sha']
model=YOLO(str(weights));assert model.names=={0:'utensil'} and model.model.model[-1].nc==1
start=time.monotonic();metrics=model.val(data=str(data/'utensil.yaml'),classes=[0],imgsz=640,batch=1,device='mps',workers=0,augment=True,save_json=True,plots=False,project=str(run/'evaluations'),name='saved-model-verification')
assert list(metrics.box.ap_class_index)==[0]
p,r,a,b=metrics.box.class_result(0)
report=dict(weights=str(weights),weights_sha=s.sha(weights),target='utensil',classes=[0],model='YOLO11x with exact COCO fork/knife/spoon maximum represented as a native one-class head',pretrained_only=True,imgsz=640,test_time_augmentation=True,nms_iou=.7,max_det=300,metric_min_confidence=.001,device='mps',half=False,dataset_fingerprint=m['fingerprint'],validation_images=764,validation_utensil_boxes=296,validation_unchanged=True,independent_test=False,precision=float(p),recall=float(r),AP50=float(a),AP50_95=float(b),elapsed_seconds=time.monotonic()-start,goal_90pct_AP50_met=float(a)>=.90,raw_metrics=metrics.results_dict)
s.save(run/'verification.json',report)
reference=s.load(root/'model/utensil-focus/pretrained-union-v13/yolo11x-tta640.json')
assert abs(report['AP50']-reference['AP50'])<.002,('Standalone conversion changed AP50',report['AP50'],reference['AP50'])
assert s.sha(weights)==initial['weights_sha']
s.validate(m['records'],data)
base=s.load(root/'data/utensil-packet-focus/meal-scenes-v10/manifest.json')
assert [(x['name'],x['sha'],[r for r in x['rows'] if r.startswith('0 ')]) for x in m['records'] if x['split']=='val']==[(x['name'],x['sha'],[r for r in x['rows'] if r.startswith('0 ')]) for x in base['records'] if x['split']=='val']
s.save(run/'inference.json',dict(weights='utensil-yolo11x.pt',weights_sha=report['weights_sha'],classes=[0],names=['utensil'],imgsz=640,augment=True,iou=.7,max_det=300,half=False,validated_AP50=report['AP50'],validation_report='verification.json',independent_test=False))
print('VERIFIED_UTENSIL_90',json.dumps(report),flush=True)
if not report['goal_90pct_AP50_met']:raise RuntimeError('Verified saved model did not meet 90% AP50; continue experiments')
