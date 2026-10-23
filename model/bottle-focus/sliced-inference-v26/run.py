"""Evaluate the official SAHI implementation through the unchanged native validator."""
from pathlib import Path
import importlib.util,json,time
import torch
from PIL import Image,ImageOps
from ultralytics import YOLO,__version__ as ultra_version
from ultralytics.models.yolo.detect import DetectionValidator
import sahi
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.slicing import get_slice_bboxes
from adapter import predictions_for_validation

RUN=Path(__file__).resolve().parent
torch.set_num_threads(2)
spec=importlib.util.spec_from_file_location('verified_training_data',RUN.parent/'context-balance-v20/build.py')
dataset=importlib.util.module_from_spec(spec);spec.loader.exec_module(dataset)
audit=dataset.verify();m=dataset.m;s=m.s
weights=RUN.parent/'context-head-soup-v23/alpha0.5-bottle.pt'
proof=s.load(RUN.parent/'context-head-soup-v23/alpha0.5-ordinary640-iou0.5-cpu.json')
before=s.sha(weights);assert before==proof['weights_sha']=='2cd1efe2d7c187df05ed223ba2a18e147393358179f223c264663a2114812289'
control=s.load(RUN.parent/'context-head-soup-v23/ordinary-batch1-iou0.5-mps.json')
assert control['weights_sha']==before and control['batch']==1 and control['device']=='mps' and control['nms_iou']==.5
geometry=s.load(RUN/'geometry-verification.json')
assert geometry['tests_passed']==2 and geometry['adapter_sha']==s.sha(RUN/'adapter.py') and geometry['test_sha']==s.sha(RUN/'test_adapter.py')
assert sahi.__version__=='0.12.8'
detector=AutoDetectionModel.from_pretrained(model_type='ultralytics',model_path=str(weights),confidence_threshold=.001,device='mps',image_size=640)
detector.model.overrides.update(iou=.5,max_det=300,augment=False,classes=[0],half=False)
trace=dict(images=0,slice_predictions=0,additional_full_predictions=0,original_shapes=set())

class SlicedValidator(DetectionValidator):
    def preprocess(self,batch):
        self.source_batch=batch
        return super().preprocess(batch)

    def postprocess(self,unused_native_predictions):
        batch=self.source_batch
        assert len(batch['im_file'])==1
        with Image.open(batch['im_file'][0]) as opened:
            image=ImageOps.exif_transpose(opened).convert('RGB')
        assert (image.height,image.width)==tuple(batch['ori_shape'][0])
        count=len(get_slice_bboxes(image_height=image.height,image_width=image.width,slice_height=384,slice_width=384,overlap_height_ratio=.25,overlap_width_ratio=.25,auto_slice_resolution=False))
        result=get_sliced_prediction(image,detector,slice_height=384,slice_width=384,overlap_height_ratio=.25,overlap_width_ratio=.25,perform_standard_pred=True,postprocess_type='NMS',postprocess_match_metric='IOU',postprocess_match_threshold=.5,postprocess_class_agnostic=False,verbose=0,auto_slice_resolution=False,batch_size=1,force_postprocess_type=True,progress_bar=False)
        trace['images']+=1;trace['slice_predictions']+=count;trace['additional_full_predictions']+=int(count>1)
        trace['original_shapes'].add((image.height,image.width))
        return [predictions_for_validation(result.object_prediction_list,batch['ratio_pad'][0],self.device,max_det=300)]

started=time.monotonic()
metrics=YOLO(str(weights)).val(validator=SlicedValidator,data=str(m.DATA/'data.yaml'),classes=[0],imgsz=640,batch=1,device='mps',workers=0,augment=False,iou=.5,conf=.001,max_det=300,save_json=True,plots=False,project=str(RUN/'evaluations'),name='sahi384-overlap25-full-nms0.5')
assert list(metrics.box.ap_class_index)==[0] and int(metrics.nt_per_class[0])==67 and trace['images']==246
p,r,ap,ap95=metrics.box.class_result(0)
report=dict(weights=str(weights),weights_sha=before,AP50=float(ap),AP50_95=float(ap95),precision=float(p),recall=float(r),device='mps',imgsz=640,batch=1,nms_iou=.5,confidence_threshold=.001,max_det=300,test_time_augmentation=True,inference_pipeline='Official SAHI on original-resolution RGB images; 384x384 slices, 25% overlap, full-image prediction and class-aware NMS',slice_height=384,slice_width=384,overlap_ratio=.25,perform_standard_pred=True,sahi_version=sahi.__version__,ultralytics_version=ultra_version,adapter_sha=s.sha(RUN/'adapter.py'),runner_sha=s.sha(Path(__file__)),geometry_verification=geometry,validation_images=246,bottle_objects=67,source_dataset_fingerprint=m.MANIFEST['source_fingerprint'],validation_protocol=m.MANIFEST['validation_protocol'],independent_test=False,ordinary_reference=proof,matched_ordinary_control=control,elapsed_seconds=time.monotonic()-started,source_images_read=trace['images'],slice_predictions=trace['slice_predictions'],additional_full_predictions=trace['additional_full_predictions'],original_shapes=sorted(trace['original_shapes']),evaluation_overhead='The native validator also performs an unused ordinary forward pass per image; included in elapsed time.')
assert s.sha(weights)==before;dataset.verify();s.save(RUN/'verification.json',report)
print('SLICED_BOTTLE_RESULT',json.dumps(report),flush=True)
