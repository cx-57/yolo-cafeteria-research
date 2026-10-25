from pathlib import Path
import importlib.util
import unittest
p=Path(__file__).with_name('protocol.py');s=importlib.util.spec_from_file_location('protocol42',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class ProtocolTests(unittest.TestCase):
 def test_missing_or_wrong_authorization_rejected(self):
  for receipt in [None,{},dict(declaration_sha='other',full_training_authorized=True,user_instruction='yes')]:
   with self.assertRaises(ValueError):m.validate_authorization(receipt,'correct')
  m.validate_authorization(dict(declaration_sha='correct',full_training_authorized=True,user_instruction='Start full training'),'correct')
 def test_boolean_and_text_authorization_required(self):
  for value,text in [(1,'yes'),(True,''),(True,'   ')]:
   with self.assertRaises(ValueError):m.validate_authorization(dict(declaration_sha='correct',full_training_authorized=value,user_instruction=text),'correct')
 def test_one_source_regression_blocks_gate(self):
  self.assertFalse(m.eligible([.8,.6,.49,.3],[.7,.5,.5,.28]))
 def test_mean_gain_and_all_metrics_required(self):
  self.assertTrue(m.eligible([.72,.52,.52,.30],[.7,.5,.5,.28]))
  self.assertFalse(m.eligible([.7,.5,.5,.28],[.7,.5,.5,.28]))
 def test_invalid_metric_pairs_rejected(self):
  for values in [[float('nan'),.5,.5,.2],[.5,.6,.5,.2],[.6,.4,1.1,.4]]:
   with self.assertRaises(ValueError):m.rank(values,[.7,.5,.5,.28])
 def test_worst_source_improvement_ranks_first(self):
  baseline=[.7,.5,.5,.28]
  self.assertGreater(m.rank([.72,.52,.52,.3],baseline),m.rank([.85,.65,.51,.29],baseline))
if __name__=='__main__':unittest.main()
