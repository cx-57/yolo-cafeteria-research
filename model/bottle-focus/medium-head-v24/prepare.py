"""Prepare a COCO-initialized YOLO11m bottle specialist with output identity checks."""
from pathlib import Path
import copy,hashlib,json
import torch
from ultralytics import YOLO,__version__
RUN=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
proof=json.loads((RUN/'download.json').read_text());source=Path(proof['source'])
assert sha(source)==proof['source_sha']
torch.set_num_threads(2);original=YOLO(str(source)).model.float().cpu().eval()
assert original.names[39]=='bottle' and original.model[-1].nc==80
assert original.yaml['scale']=='m'
projected=copy.deepcopy(original);head=projected.model[-1]
for branch in head.cv3:
    old=branch[-1];new=torch.nn.Conv2d(old.in_channels,1,old.kernel_size,old.stride,old.padding,bias=old.bias is not None)
    with torch.no_grad():
        new.weight.copy_(old.weight[39:40])
        if old.bias is not None:new.bias.copy_(old.bias[39:40])
    branch[-1]=new
head.nc=1;head.no=4*head.reg_max+1;projected.yaml['nc']=1;projected.names={0:'bottle'};projected.criterion=None
with torch.inference_mode():
    torch.manual_seed(42);x=torch.rand(1,3,96,128);a,b=original(x)[0],projected(x)[0]
    torch.testing.assert_close(a[:,:4],b[:,:4],rtol=0,atol=0)
    torch.testing.assert_close(a[:,43:44],b[:,4:],rtol=1e-6,atol=1e-7)
assert sha(source)==proof['source_sha']
initial=RUN/'initial.pt';assert not initial.exists()
torch.save(dict(model=projected.half(),ema=None,optimizer=None,epoch=-1,version=__version__,train_args=dict(task='detect')),initial)
result=dict(source=str(source),source_sha=sha(source),weights_sha=sha(initial),source_url=proof['url'],COCO_class=39,architecture='YOLO11m',single_class_parameters=sum(p.numel() for p in projected.parameters()),decoded_boxes_preserved_exactly=True,COCO_bottle_scores_preserved=True,probe_shape=list(x.shape),ultralytics_version=__version__)
(RUN/'initialization.json').write_text(json.dumps(result,indent=2)+'\n');print('MEDIUM_BOTTLE_INITIALIZED',json.dumps(result),flush=True)
