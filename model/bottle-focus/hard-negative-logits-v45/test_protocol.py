from pathlib import Path
import importlib.util
import unittest
p=Path(__file__).with_name('protocol.py');spec=importlib.util.spec_from_file_location('pr44test',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
BASE=[.9,.7,.62,.51,.48,.28]
class ProtocolTests(unittest.TestCase):
 def test_source_gains_cannot_hide_retention_regression(self):
  self.assertFalse(m.eligible([.89,.7,.9,.7,.7,.4],BASE))
 def test_any_old_source_regression_blocks(self):
  self.assertFalse(m.eligible([.91,.71,.8,.6,.47,.29],BASE))
 def test_gains_required_and_small_retention_tolerance_explicit(self):
  self.assertTrue(m.eligible([.897,.698,.7,.6,.6,.35],BASE));self.assertFalse(m.eligible(BASE,BASE))
 def test_no_eligible_candidate_keeps_initial(self):
  initial=dict(role='initial',values=BASE,epoch=0);bad=dict(role='latest',values=[.8,.6,.9,.7,.8,.6],epoch=6)
  self.assertEqual(m.choose([initial,bad],BASE),initial)
 def test_invalid_values_rejected(self):
  for values in [[.9,.95,.6,.4,.5,.3],[float('nan'),.7,.6,.4,.5,.3]]:
   with self.assertRaises(ValueError):m.eligible(values,BASE)
 def test_saved_score_identity_rejects_changed_model_or_support(self):
  expected=dict(weights_sha='abc',photos=24,boxes=18);report=dict(expected,AP50=.9,AP50_95=.7)
  m.receipt(report,expected)
  for altered in [dict(expected,weights_sha='other'),dict(expected,boxes=19)]:
   with self.assertRaises(AssertionError):m.receipt(report,altered)
if __name__=='__main__':unittest.main()
