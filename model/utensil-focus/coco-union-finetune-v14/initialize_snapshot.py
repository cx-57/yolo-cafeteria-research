from pathlib import Path
import copy,json,hashlib
import torch,yaml
from ultralytics import YOLO,__version__
import utensil_packet as s
root=Path.cwd();run=root/'model/utensil-focus/coco-union-finetune-v14';run.mkdir(exist_ok=False)
source=root/'model/weights/yolo11s.pt';original=YOLO(str(source)).model.float().cpu().eval();projected=copy.deepcopy(original);head=projected.model[-1]
assert head.nc==80 and not head.end2end
for branch in head.cv3:
 old=branch[-1];new=torch.nn.Conv2d(old.in_channels,3,old.kernel_size,old.stride,old.padding,bias=old.bias is not None)
 with torch.no_grad():
  new.weight.copy_(old.weight[[42,43,44]])
  if old.bias is not None:new.bias.copy_(old.bias[[42,43,44]])
 branch[-1]=torch.nn.Sequential(new,torch.nn.MaxPool3d((3,1,1)))
head.nc=1;head.no=4*head.reg_max+1;projected.yaml['nc']=1;projected.names={0:'utensil'};projected.criterion=None
with torch.no_grad():
 torch.set_num_threads(2);torch.manual_seed(42);x=torch.rand(2,3,96,96);a=original(x)[0];b=projected(x)[0]
 torch.testing.assert_close(a[:,:4],b[:,:4],rtol=0,atol=0)
 torch.testing.assert_close(a[:,46:49].amax(dim=1,keepdim=True),b[:,4:],rtol=1e-6,atol=1e-7)
# Native pooling uses 4D unbatched C,D,H,W: D is the three logits, C the untouched batch.
x=torch.randn(2,3,8,9,device='mps',requires_grad=True);out=torch.nn.MaxPool3d((3,1,1))(x);torch.testing.assert_close(out,x.amax(dim=1,keepdim=True),rtol=0,atol=0);out.sum().backward();assert torch.isfinite(x.grad).all()
weights=run/'initial.pt';torch.save(dict(model=projected.half(),ema=None,optimizer=None,epoch=-1,version=__version__,train_args=dict(task='detect',epochs=40)),weights)
data=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/utensil-v12');config=yaml.safe_load((data/'data.yaml').read_text());config['names']=['utensil'];(data/'utensil.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
s.save(run/'initialization.json',dict(source=str(source),source_sha=s.sha(source),weights_sha=s.sha(weights),preserved_COCO_classes=[42,43,44],decoded_boxes_identical=True,union_scores_identical=True,MPS_pooling_forward_and_backward_verified=True,architecture='Original YOLO11s with three retained COCO utensil logits reduced by exact maximum before the native one-class detection loss. No custom Python layers.',dataset_fingerprint=s.load(data/'manifest.json')['fingerprint'],independent_test=False))
print('Verified exact decoded-box and union-score equivalence; MPS gradients finite. Saved single-class initialization.',flush=True)
