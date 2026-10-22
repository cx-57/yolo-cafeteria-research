from pathlib import Path
import copy,json,gc,time
import torch
from ultralytics import YOLO,__version__
import utensil_packet as s
root=Path.cwd();run=root/'model/bottle-focus/validation-now';run.mkdir(parents=True,exist_ok=False);data=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-capture-validation');manifest=s.load(data/'manifest.json');assert manifest['validation_images']==246 and manifest['validation_bottle_objects']==67
source=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/weights/yolo11x.pt');original=YOLO(str(source)).model.float().cpu().eval();assert original.names[39]=='bottle';projected=copy.deepcopy(original);head=projected.model[-1]
for branch in head.cv3:
 old=branch[-1];new=torch.nn.Conv2d(old.in_channels,1,old.kernel_size,old.stride,old.padding,bias=old.bias is not None)
 with torch.no_grad():
  new.weight.copy_(old.weight[39:40])
  if old.bias is not None:new.bias.copy_(old.bias[39:40])
 branch[-1]=new
head.nc=1;head.no=4*head.reg_max+1;projected.yaml['nc']=1;projected.names={0:'bottle'};projected.criterion=None
with torch.no_grad():
 torch.set_num_threads(2);x=torch.rand(1,3,96,96);a=original(x)[0];b=projected(x)[0];torch.testing.assert_close(a[:,:4],b[:,:4],rtol=0,atol=0);torch.testing.assert_close(a[:,43:44],b[:,4:],rtol=1e-6,atol=1e-7)
weights=run/'bottle-yolo11x.pt';torch.save(dict(model=projected.half(),ema=None,optimizer=None,epoch=-1,version=__version__,train_args=dict(task='detect')),weights)
s.save(run/'initialization.json',dict(source=str(source),source_sha=s.sha(source),weights_sha=s.sha(weights),COCO_class=39,decoded_boxes_and_bottle_scores_preserved=True,pretrained_only=True))
del original,projected,head,a,b,x;gc.collect();torch.mps.empty_cache()
s.NAMES=['bottle'];s.validate_configuration(data,manifest['records']);s.validate(manifest['records'],data)
model=YOLO(str(weights));start=time.monotonic();metrics=model.val(data=str(data/'data.yaml'),split='val',classes=[0],imgsz=640,batch=1,device='mps',workers=0,augment=True,save_json=True,plots=False,project=str(run),name='evaluation')
assert list(metrics.box.ap_class_index)==[0] and int(metrics.nt_per_class[0])==67
p,r,a,b=metrics.box.class_result(0);report=dict(target='bottle',model='YOLO11x pretrained COCO bottle class projected to one output',pretrained_only=True,weights=str(weights),weights_sha=s.sha(weights),split='val',source_dataset=manifest['source'],source_dataset_fingerprint=manifest['source_fingerprint'],validation_protocol=manifest['validation_protocol'],validation_images=246,bottle_objects=67,classes=[0],imgsz=640,test_time_augmentation=True,precision=float(p),recall=float(r),AP50=float(a),AP50_95=float(b),independent_test=False,elapsed_seconds=time.monotonic()-start)
s.validate(manifest['records'],data);s.save(run/'verification.json',report);print('BOTTLE_FRESH_VALIDATION',json.dumps(report),flush=True)
