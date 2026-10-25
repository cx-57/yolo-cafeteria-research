"""Classifier-only optimizer and normalization/EMA protection for YOLO11s."""
import torch
from ultralytics.models.yolo.detect import DetectionTrainer
import geometry
class Trainer(DetectionTrainer):
 def build_optimizer(self,model,name='auto',lr=.001,momentum=.9,decay=1e-5,iterations=1e5):
  assert name=='AdamW'
  optimizer=geometry.optimizer(model,lr,momentum,decay);self.geometry_reference=geometry.fixed_state(model)
  return optimizer
 def _model_train(self):
  super()._model_train()
  for module in self.model.modules():
   if isinstance(module,torch.nn.BatchNorm2d):module.eval()
 def optimizer_step(self):
  super().optimizer_step()
  if self.ema:geometry.restore_ema(self.ema.ema,self.model)
 def final_eval(self):pass
