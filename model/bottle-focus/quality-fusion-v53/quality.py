"""Fixed DFL top-two concentration heuristic; no learned quality predictor."""
import types
import torch

def certainty(logits,regmax):
 assert logits.ndim==3 and regmax>=2 and logits.shape[1]==4*regmax and torch.isfinite(logits).all()
 b,_,n=logits.shape
 mass=logits.reshape(b,4,regmax,n).softmax(2).topk(2,dim=2).values.sum(2).mean(1,keepdim=True)
 return mass.clamp(0,1).sqrt()

def install(model):
 head=model.model[-1];assert not head.end2end and head.nc==1 and head.reg_max==16
 original=type(head)._inference;trace=dict(calls=0)
 def adjusted(self,raw):
  out=original(self,raw);weight=certainty(raw['boxes'],self.reg_max);assert weight.shape==(out.shape[0],1,out.shape[2])
  trace['calls']+=1
  return torch.cat((out[:,:4],out[:,4:]*weight),1)
 head._inference=types.MethodType(adjusted,head);return trace
