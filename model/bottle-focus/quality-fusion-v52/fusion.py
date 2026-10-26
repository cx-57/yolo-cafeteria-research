"""Combine original and horizontal-flip views using standard weighted box fusion."""
import torch
from ultralytics.models.yolo.detect import DetectionValidator
from vendor_wbf import weighted_boxes_fusion

def fuse_views(views,width,height):
    assert len(views)==2 and width>0 and height>0
    reference=views[0]['bboxes'];scale=reference.new_tensor([width,height,width,height])
    boxes_list=[];scores_list=[];labels_list=[]
    for view in views:
        boxes=view['bboxes'];scores=view['conf'];labels=view['cls']
        assert boxes.shape==(len(scores),4) and len(labels)==len(scores)
        assert torch.isfinite(boxes).all() and torch.isfinite(scores).all()
        assert ((scores>=0)&(scores<=1)).all() and (labels==0).all()
        normalized=(boxes/scale).clamp(0,1)
        valid=(normalized[:,2]>normalized[:,0])&(normalized[:,3]>normalized[:,1])
        boxes_list.append(normalized[valid].cpu().numpy())
        scores_list.append(scores[valid].cpu().numpy())
        labels_list.append(labels[valid].cpu().numpy())
    boxes,scores,labels=weighted_boxes_fusion(boxes_list,scores_list,labels_list,weights=[1,1],iou_thr=.55,skip_box_thr=.001,conf_type='avg',allows_overflow=False)
    result=dict(bboxes=torch.as_tensor(boxes,dtype=reference.dtype,device=reference.device)*scale,
                conf=torch.as_tensor(scores,dtype=reference.dtype,device=reference.device),
                cls=torch.as_tensor(labels,dtype=reference.dtype,device=reference.device),
                extra=reference.new_empty((len(scores),0)))
    return result

class FlipFusionValidator(DetectionValidator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.frame_hw=None
        self.fused_images=0

    def preprocess(self,batch):
        self.frame_hw=tuple(batch['img'].shape[-2:])
        return super().preprocess(batch)

    def postprocess(self,preds):
        raw=preds[0] if isinstance(preds,(tuple,list)) else preds
        assert raw.ndim==3 and raw.shape[1]==5 and raw.shape[-1]%2==0
        assert self.frame_hw is not None and not self.end2end
        height,width=self.frame_hw
        parent=super().postprocess
        per_view=[parent(chunk.clone()) for chunk in raw.chunk(2,-1)]
        outputs=[]
        for index in range(raw.shape[0]):
            fused=fuse_views([view[index] for view in per_view],width,height)
            outputs.append({key:value[:self.args.max_det] for key,value in fused.items()})
        self.fused_images+=len(outputs)
        return outputs
