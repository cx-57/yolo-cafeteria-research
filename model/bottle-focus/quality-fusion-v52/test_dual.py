import copy
import unittest
import torch
from ultralytics import YOLO
from dual import attach
BASE='/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-reviewed-neck-monitor-v36/weights/verified-88877bbf13c02a202811b6ca0a1ccdc2bde70b2916d366f4b67938df732d1518.pt'
OTHER='/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-reviewed-full-v42/weights/verified-95370c9b321b57feebbf11204b51a7a8393085dd80bb0ae483b0a1a1f2433013.pt'
class DualTests(unittest.TestCase):
 def test_actual_models_match_each_segment_and_register_secondary(self):
  torch.set_num_threads(2);a=YOLO(BASE).model.float().cpu().eval();b=YOLO(OTHER).model.float().cpu().eval();x=torch.rand(2,3,96,128)
  with torch.no_grad():left=a(x)[0].clone();right=b(x)[0].clone()
  trace=attach(a,b)
  with torch.no_grad():out=a(x,augment=True)[0]
  torch.testing.assert_close(out,torch.cat([left,right],-1),rtol=0,atol=0)
  self.assertEqual(trace['calls'],1);self.assertIn('secondary',a._modules)
  self.assertEqual(trace['shapes'],[(2,3,96,128)])
  a.double();self.assertEqual(next(a.secondary.parameters()).dtype,torch.float64)
if __name__=='__main__':unittest.main()
