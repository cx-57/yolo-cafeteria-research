"""Reject accidental backbone updates, head-only training or changing BN statistics."""
import torch

def audit(model):
    trainable=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
    indices=sorted({int(name.split('.')[1]) for name,p in trainable})
    assert indices and min(indices)>=11 and 23 in indices and any(i<23 for i in indices)
    normalization=[layer for layer in model.modules() if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm)]
    assert normalization and not any(layer.training for layer in normalization)
    return dict(backbone_parameters_fixed=True,neck_and_head_trainable=True,trainable_layer_indices=indices,trainable_parameters=sum(p.numel() for name,p in trainable),all_BatchNorm_statistics_fixed=True,normalization_modules=len(normalization))
