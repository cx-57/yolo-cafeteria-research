"""Shrink only the declared score tensors; reject any protected endpoint drift."""
import math
import torch

def blend_states(base,endpoint,allowed,alpha):
 assert type(alpha) in (int,float) and math.isfinite(alpha) and 0<=alpha<=1
 assert set(base)==set(endpoint) and allowed and set(allowed)<=set(base)
 result={}
 for name,left in base.items():
  right=endpoint[name]
  assert left.dtype==right.dtype and left.shape==right.shape
  assert torch.isfinite(left).all() and torch.isfinite(right).all()
  if name in allowed:
   assert left.is_floating_point()
   result[name]=left.clone() if alpha==0 else right.clone() if alpha==1 else left+(right-left)*alpha
  else:
   assert torch.equal(left,right),'Protected endpoint tensor changed: '+name
   result[name]=left.clone()
 return result
