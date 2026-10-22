from pathlib import Path
import json,types,gc
import torch
from ultralytics import YOLO
import utensil_packet as s
root=Path.cwd();run=root/'model/utensil-focus/pretrained-union-v13';run.mkdir(exist_ok=True)
for size,augment in [('n',False),('s',False),('s',True)]:
 tag=f'yolo11{size}-'+('tta640' if augment else '640');dest=run/(tag+'.json')
 if dest.exists():continue
 weights=root/f'model/weights/yolo11{size}.pt';model=YOLO(str(weights));head=model.model.model[-1]
 assert head.nc==80
 assert [model.names[i] for i in [42,43,44]]==['fork','knife','spoon']
 original_forward=model.model.forward
 def union_forward(self,x,*args,**kwargs):
  assert x.device == next(self.parameters()).device
  output=type(self).forward(self,x,*args,**kwargs)
  pred=output[0] if isinstance(output,tuple) else output
  merged=pred[:,46:49].amax(dim=1,keepdim=True)
  result=torch.cat((pred[:,:4],merged,torch.zeros_like(merged)),dim=1)
  return (result,output[1]) if isinstance(output,tuple) and len(output)>1 else result
 model.model.forward=types.MethodType(union_forward,model.model)
 model.model.names={0:'utensil',1:'condiment_packet'}
 metrics=model.val(data=str(root/'data/utensil-only/coverage-v12/data.yaml'),classes=[0],imgsz=640,batch=1 if augment else 4,device='mps',workers=0,augment=augment,plots=False,project=str(run/'evaluations'),name=tag)
 assert list(metrics.box.ap_class_index)==[0]
 p,r,a,b=metrics.box.class_result(0)
 report=dict(weights=str(weights),weights_sha=s.sha(weights),target='utensil',mapping='COCO fork/knife/spoon merged by per-anchor maximum before NMS; packet score always zero.',classes=[0],imgsz=640,test_time_augmentation=augment,pretrained_only=True,independent_test=False,validation_images=764,validation_utensil_boxes=296,precision=float(p),recall=float(r),AP50=float(a),AP50_95=float(b))
 s.save(dest,report);print('UNION_RESULT',tag,json.dumps(report),flush=True)
 if size=='n':
  expected=s.load(root/'model/utensil-packet-focus/pretrained-utensil-control.json')['utensil']['AP50']
  assert abs(float(a)-expected)<.002,('Prior diagnostic failed to reproduce',float(a),expected)
 del model;gc.collect();torch.mps.empty_cache()
