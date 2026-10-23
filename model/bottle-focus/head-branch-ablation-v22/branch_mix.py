"""Build independent detection-head branch comparisons."""
import copy
import torch

def mix_branches(initial,quality,branch):
    assert branch in ('regression','classification')
    a,b=initial.state_dict(),quality.state_dict();assert a.keys()==b.keys()
    for key,value in a.items():
        if not torch.equal(value,b[key]):
            assert key.startswith(('model.23.cv2.','model.23.cv3.')),key
            assert not any(t in key for t in ('running_mean','running_var','num_batches_tracked')),key
    prefix='model.23.cv2.' if branch=='regression' else 'model.23.cv3.'
    mixed=copy.deepcopy(initial);state=mixed.state_dict();changed=[]
    for key,value in state.items():
        if key.startswith(prefix):
            if not torch.equal(value,b[key]):changed.append(key)
            value.copy_(b[key])
    assert changed
    mixed.load_state_dict(state,strict=True)
    return mixed
