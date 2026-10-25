"""Train classification parameters only; protect all decoded geometry and buffers."""
import hashlib
import torch
PREFIX='model.23.cv3.'
def select(model):
 names=[]
 assert getattr(model,'names',None)=={0:'bottle'} and model.model[-1].nc==1,'Unsupported head'
 for name,param in model.named_parameters():
  active=name.startswith(PREFIX);param.requires_grad_(active)
  if active:names.append(name)
 assert names and any(n.startswith('model.23.cv2.') for n,_ in model.named_parameters())
 for module in model.modules():
  if isinstance(module,torch.nn.BatchNorm2d):module.eval()
 return set(names)
def fixed_state(model):
 allowed={n for n,_ in model.named_parameters() if n.startswith(PREFIX)}
 return {n:t.detach().cpu().clone() for n,t in model.state_dict().items() if n not in allowed}
def verify_fixed(model,reference):
 state=model.state_dict();assert set(reference)==set(fixed_state(model))
 for name,value in reference.items():assert torch.equal(state[name].detach().cpu(),value),'Fixed tensor changed: '+name
 return True
def restore_ema(ema,model):
 allowed={n for n,_ in model.named_parameters() if n.startswith(PREFIX)};source=model.state_dict()
 with torch.no_grad():
  for name,value in ema.state_dict().items():
   if name not in allowed:value.copy_(source[name])
def optimizer(model,lr,momentum,decay):
 select(model);groups={'bias':[],'weight':[],'bn':[]}
 for module_name,module in model.named_modules():
  for name,p in module.named_parameters(recurse=False):
   if not p.requires_grad:continue
   role='bias' if name=='bias' else 'bn' if isinstance(module,torch.nn.modules.batchnorm._BatchNorm) else 'weight'
   groups[role].append(p)
 assert all(groups.values())
 return torch.optim.AdamW([dict(params=groups[role],weight_decay=decay if role=='weight' else 0.,param_group=role) for role in ['bias','weight','bn']],lr=lr,betas=(momentum,.999),weight_decay=0.)
def fingerprint(reference):
 digest=hashlib.sha256()
 for name,value in sorted(reference.items()):
  digest.update(name.encode());digest.update(str(value.dtype).encode());digest.update(str(tuple(value.shape)).encode());digest.update(value.numpy().tobytes())
 return digest.hexdigest()
