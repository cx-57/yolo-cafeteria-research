import importlib.util
from pathlib import Path
import unittest
from ultralytics import YOLO

RUN=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v36_trainability',RUN/'trainability.py');policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)

class TrainabilityTests(unittest.TestCase):
 def model(self,freeze):
  model=YOLO(str(RUN.parent/'quality-head-blend-v19/alpha0.25-bottle.pt')).model.cpu().eval()
  for name,p in model.named_parameters():p.requires_grad_(int(name.split('.')[1])>=freeze and '.dfl.' not in name)
  return model
 def test_real_yolo11_backbone_including_c2psa_fixed_neck_and_head_train(self):
  report=policy.validate(self.model(11))
  self.assertTrue(report['full_backbone_fixed']);self.assertGreater(report['neck_parameters'],0)
 def test_freeze10_would_expose_c2psa_and_is_rejected(self):
  with self.assertRaises(AssertionError):policy.validate(self.model(10))
if __name__=='__main__':unittest.main()
