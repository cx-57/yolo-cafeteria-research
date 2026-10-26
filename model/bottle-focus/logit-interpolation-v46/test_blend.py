import unittest
import torch
from blend import blend_states

class BlendTests(unittest.TestCase):
 def test_only_allowed_changes_and_endpoints(self):
  base={'score':torch.tensor([1.,3.]),'box':torch.tensor([4.]),'count':torch.tensor(2)}
  endpoint={k:v.clone() for k,v in base.items()};endpoint['score']+=4
  out=blend_states(base,endpoint,{'score'},.25)
  self.assertTrue(torch.equal(out['score'],torch.tensor([2.,4.])))
  self.assertTrue(torch.equal(out['box'],base['box']))
  self.assertTrue(torch.equal(out['count'],base['count']))
  out['box'].add_(1);self.assertEqual(base['box'].item(),4)
  for a,expected in [(0,base),(1,endpoint)]:
   result=blend_states(base,endpoint,{'score'},a)
   self.assertTrue(all(torch.equal(result[k],expected[k]) for k in base))
 def test_reject_protected_drift_and_invalid_input(self):
  base={'score':torch.tensor([1.]),'box':torch.tensor([4.])}
  changed={k:v.clone() for k,v in base.items()};changed['box']+=1
  with self.assertRaises(AssertionError):blend_states(base,changed,{'score'},.5)
  for alpha in [float('nan'),-.1,1.1,True]:
   with self.assertRaises(AssertionError):blend_states(base,base,{'score'},alpha)
  with self.assertRaises(AssertionError):blend_states(base,base,{'missing'},.5)
  broken=dict(base,score=torch.tensor([float('nan')]))
  with self.assertRaises(AssertionError):blend_states(base,broken,{'score'},.5)
if __name__=='__main__':unittest.main()
