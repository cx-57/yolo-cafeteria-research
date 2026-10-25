from pathlib import Path
import importlib.util
import unittest
import copy
import torch
from ultralytics import YOLO
p=Path(__file__).with_name('geometry.py');spec=importlib.util.spec_from_file_location('geometry44test',p);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
SOURCE=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-reviewed-neck-monitor-v36/weights/verified-88877bbf13c02a202811b6ca0a1ccdc2bde70b2916d366f4b67938df732d1518.pt')
class GeometryTests(unittest.TestCase):
 def test_real_classifier_step_changes_scores_but_preserves_all_boxes_and_fixed_tensors(self):
  torch.set_num_threads(2);torch.manual_seed(43);model=YOLO(str(SOURCE)).model.float().cpu().eval();names=g.select(model);reference=g.fixed_state(model);ema=copy.deepcopy(model)
  self.assertEqual(len(names),6);self.assertEqual(sum(p.numel() for n,p in model.named_parameters() if n in names),387)
  x=torch.rand(2,3,96,96)
  before=model(x)[0].detach().clone();optimizer=g.optimizer(model,1e-3,.9,.0005)
  actual={id(p) for group in optimizer.param_groups for p in group['params']}
  self.assertEqual(actual,{id(p) for p in model.parameters() if p.requires_grad})
  optimizer.zero_grad();output=model(x)[0];output[:,4:].mean().backward();optimizer.step();after=model(x)[0].detach()
  torch.testing.assert_close(before[:,:4],after[:,:4],rtol=0,atol=0)
  self.assertFalse(torch.equal(before[:,4:],after[:,4:]));g.verify_fixed(model,reference)
  with torch.no_grad():
   for tensor in ema.state_dict().values():
    if tensor.is_floating_point():tensor.mul_(.999).add_(.00001)
  g.restore_ema(ema,model);g.verify_fixed(ema,reference)
  torch.testing.assert_close(before[:,:4],ema(x)[0][:,:4],rtol=0,atol=0)
 def test_changed_fixed_parameter_rejected(self):
  model=YOLO(str(SOURCE)).model.float().cpu().eval();g.select(model);reference=g.fixed_state(model)
  with torch.no_grad():next(model.parameters()).add_(.01)
  with self.assertRaises(AssertionError):g.verify_fixed(model,reference)
 def test_unsupported_head_fails_closed(self):
  with self.assertRaises(AssertionError):g.select(torch.nn.Linear(2,1))
if __name__=='__main__':unittest.main()
