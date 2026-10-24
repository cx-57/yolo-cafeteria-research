"""Check actual trainable YOLO11s indices; layer11 itself is parameter-free."""
import torch
def validate(model):
 trainable=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
 assert trainable and all(int(name.split('.')[1])>=11 for name,p in trainable)
 neck=sum(p.numel() for name,p in trainable if 11<=int(name.split('.')[1])<23)
 head=sum(p.numel() for name,p in trainable if int(name.split('.')[1])==23)
 assert neck>0 and head>0
 norms=[layer for layer in model.modules() if isinstance(layer,torch.nn.BatchNorm2d)]
 assert norms and not any(layer.training for layer in norms)
 return dict(full_backbone_fixed=True,neck_and_head_only=True,all_BatchNorm_statistics_fixed=True,
             normalization_modules=len(norms),neck_parameters=neck,head_parameters=head)
