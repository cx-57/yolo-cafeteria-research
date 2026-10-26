"""Register a frozen second detector; concatenate ordinary outputs on identical input."""
import types
import torch

def attach(primary,secondary):
 assert primary.names==secondary.names=={0:'bottle'}
 assert not primary.end2end and not secondary.end2end
 assert torch.equal(primary.stride,secondary.stride)
 primary.add_module('secondary',secondary);trace=dict(calls=0,shapes=[])
 def predict_pair(self,x):
  first=self._predict_once(x);other=self.secondary(x,augment=False)
  a=first[0] if isinstance(first,(tuple,list)) else first;b=other[0] if isinstance(other,(tuple,list)) else other
  assert a.shape==b.shape and a.ndim==3 and a.shape[1]==5
  trace['calls']+=1;trace['shapes'].append(tuple(x.shape))
  return torch.cat([a,b],-1),None
 primary._predict_augment=types.MethodType(predict_pair,primary)
 for p in primary.parameters():p.requires_grad_(False)
 return trace
