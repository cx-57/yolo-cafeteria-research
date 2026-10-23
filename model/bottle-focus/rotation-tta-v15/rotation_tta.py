"""Right-angle box inversion and full-resolution orientation views."""
import torch

def undo_rotation(prediction,k,height,width):
    assert k in range(4)
    restored=prediction.clone()
    if k==1:
        restored[:,0]=width-prediction[:,1]
        restored[:,1]=prediction[:,0]
    elif k==2:
        restored[:,0]=width-prediction[:,0]
        restored[:,1]=height-prediction[:,1]
    elif k==3:
        restored[:,0]=prediction[:,1]
        restored[:,1]=height-prediction[:,0]
    if k%2:
        restored[:,2]=prediction[:,3]
        restored[:,3]=prediction[:,2]
    return restored

def full_resolution_rotations(self,x):
    height,width=x.shape[-2:]
    views=[]
    for k in range(4):
        prediction=self._predict_once(torch.rot90(x,k,(-2,-1)))[0]
        views.append(undo_rotation(prediction,k,height,width))
    return torch.cat(views,-1),None
