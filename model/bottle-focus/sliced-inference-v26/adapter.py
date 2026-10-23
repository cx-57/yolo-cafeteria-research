"""Map SAHI's original-image boxes into the native validation coordinate frame."""
import torch

def predictions_for_validation(predictions, ratio_pad, device, max_det=300):
    ordered=sorted(predictions,key=lambda p:p.score.value,reverse=True)[:max_det]
    assert all(p.category.id==0 for p in ordered), 'Expected only bottle class 0'
    boxes=torch.tensor([p.bbox.to_xyxy() for p in ordered],dtype=torch.float32,device=device).reshape(-1,4)
    scores=torch.tensor([p.score.value for p in ordered],dtype=torch.float32,device=device)
    assert torch.isfinite(boxes).all() and torch.isfinite(scores).all()
    assert ((scores>=0)&(scores<=1)).all()
    (rh,rw),(dw,dh)=ratio_pad
    boxes=boxes*boxes.new_tensor([rw,rh,rw,rh])+boxes.new_tensor([dw,dh,dw,dh])
    return dict(bboxes=boxes,conf=scores,cls=torch.zeros_like(scores),extra=boxes.new_empty((len(ordered),0)))
