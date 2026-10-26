import unittest
import copy
import torch
from quality import certainty,install
from test_dual import BASE
from ultralytics import YOLO
class QualityTests(unittest.TestCase):
 def test_peaked_distribution_and_uniform(self):
  uniform=torch.zeros(1,64,3)
  torch.testing.assert_close(certainty(uniform,16),torch.full((1,1,3),(.125)**.5))
  sharp=uniform.reshape(1,4,16,3).clone();sharp[:,:,4,:]=20
  self.assertTrue(torch.all(certainty(sharp.reshape(1,64,3),16)>.999))
 def test_actual_head_preserves_boxes_and_only_rescales_scores(self):
  torch.set_num_threads(2);m=YOLO(BASE).model.float().cpu().eval();m.fuse(verbose=False);x=torch.rand(2,3,96,128)
  with torch.no_grad():before=m(x)[0].clone()
  trace=install(m)
  with torch.no_grad():after=m(x)[0]
  torch.testing.assert_close(before[:,:4],after[:,:4],rtol=0,atol=0)
  self.assertTrue(torch.all(after[:,4:]<=before[:,4:]));self.assertFalse(torch.equal(after[:,4:],before[:,4:]));self.assertEqual(trace['calls'],1)
 def test_installed_head_survives_deepcopy_and_dtype_transfer(self):
  torch.set_num_threads(2);m=YOLO(BASE).model.float().cpu().eval();install(m);clone=copy.deepcopy(m).double()
  with torch.no_grad():out=clone(torch.rand(1,3,96,128,dtype=torch.float64))[0]
  self.assertEqual(out.dtype,torch.float64)
 def test_nonfinite_logits_rejected(self):
  with self.assertRaises(AssertionError):certainty(torch.full((1,64,3),float('nan')),16)
if __name__=='__main__':unittest.main()
