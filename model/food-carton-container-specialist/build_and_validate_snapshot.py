from pathlib import Path
import copy,json,gc,time,hashlib
import torch
from ultralytics import YOLO,__version__
import utensil_packet as s
root=Path.cwd();data=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/food-carton-container-specialist');manifest=s.load(data/'manifest.json');run=root/'model/food-carton-container-specialist';run.mkdir(parents=True,exist_ok=False)
source=root/'model/runs/v2-control-n-reviewed-baseline/weights/best.pt';source_sha=s.sha(source);original=YOLO(str(source)).model.float().cpu().eval();ids=[0,4,6];names=['food','hinged_container','drink_carton'];assert [original.names[i] for i in ids]==names
projected=copy.deepcopy(original);head=projected.model[-1];assert head.nc==11
for branch in head.cv3:
 old=branch[-1];new=torch.nn.Conv2d(old.in_channels,3,old.kernel_size,old.stride,old.padding,bias=old.bias is not None)
 with torch.no_grad():
  new.weight.copy_(old.weight[ids])
  if old.bias is not None:new.bias.copy_(old.bias[ids])
 branch[-1]=new
head.nc=3;head.no=4*head.reg_max+3;projected.yaml['nc']=3;projected.names=dict(enumerate(names));projected.criterion=None
with torch.no_grad():
 torch.set_num_threads(2);torch.manual_seed(42);x=torch.rand(2,3,96,96);a=original(x)[0];b=projected(x)[0];torch.testing.assert_close(a[:,:4],b[:,:4],rtol=0,atol=0);torch.testing.assert_close(a[:,[4+i for i in ids]],b[:,4:],rtol=1e-6,atol=1e-7)
weights=run/'specialist.pt';torch.save(dict(model=projected.half(),ema=None,optimizer=None,epoch=-1,version=__version__,train_args=dict(task='detect')),weights);digest=s.sha(weights)
s.save(run/'initialization.json',dict(source=str(source),source_sha=source_sha,weights_sha=digest,original_class_ids=ids,names=names,decoded_box_and_score_equivalence_verified=True,additional_training=False,interpretation='Target-only head projected from an existing fine-tuned checkpoint; no new accuracy gain is claimed.'))
del original,projected,head,x,a,b;gc.collect();torch.mps.empty_cache()
model=YOLO(str(weights));assert model.names==dict(enumerate(names));started=time.monotonic();metrics=model.val(data=str(data/'data.yaml'),split='val',classes=[0,1,2],imgsz=640,batch=4,device='mps',workers=0,augment=False,plots=False,save_json=True,project=str(run),name='saved-model-verification')
assert list(metrics.box.ap_class_index)==[0,1,2];assert list(map(int,metrics.nt_per_class))==[840,47,371]
per_class={}
for i,name in enumerate(names):
 p,r,a,b=metrics.box.class_result(i);per_class[name]=dict(precision=float(p),recall=float(r),AP50=float(a),AP50_95=float(b),objects=int(metrics.nt_per_class[i]))
assert s.sha(weights)==digest and s.sha(source)==source_sha
for r in manifest['records']:
 image=data/'images'/r['split']/r['name'];label=data/'labels'/r['split']/Path(r['name']).with_suffix('.txt');assert hashlib.sha256(image.read_bytes()).hexdigest()==r['sha'];assert label.read_text().splitlines()==r['rows']
source_manifest=s.load(Path(manifest['source'])/'v2_build.json');assert source_manifest['dataset_sha256']==manifest['source_fingerprint']
report=dict(weights=str(weights),weights_sha=digest,source_checkpoint=str(source),source_checkpoint_sha=source_sha,split='val',names=names,per_class=per_class,overall=metrics.results_dict,validation_images=591,source_dataset=manifest['source'],source_dataset_fingerprint=manifest['source_fingerprint'],validation_images_and_target_boxes_preserved=True,train_val_identical_byte_or_pixel_overlap=False,imgsz=640,test_time_augmentation=False,additional_training=False,all_targets_at_least_85pct_AP50=all(v['AP50']>=.85 for v in per_class.values()),annotation_status=manifest['source_annotation_status'],independent_test=False,elapsed_seconds=time.monotonic()-started)
s.save(run/'verification.json',report);s.save(run/'inference.json',dict(weights='specialist.pt',weights_sha=digest,names=names,classes=[0,1,2],imgsz=640,augment=False,iou=.7,max_det=300,verification='verification.json'))
print('SPECIALIST_VERIFIED',json.dumps(report),flush=True)
