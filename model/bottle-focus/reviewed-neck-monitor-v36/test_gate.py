import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('v36_gate',Path(__file__).with_name('gate.py'));gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
class GateTests(unittest.TestCase):
 def test_monitor_gate_requires_ap50_gain_without_localization_regression(self):
  self.assertFalse(gate.eligible(dict(AP50=.80,AP50_95=.70)))
  self.assertFalse(gate.eligible(dict(AP50=.90,AP50_95=.63)))
  self.assertTrue(gate.eligible(dict(AP50=.806,AP50_95=.638)))
 def test_nonfinite_or_impossible_monitor_pair_rejected(self):
  for pair in [(float('nan'),.7),(.8,.9),(1.1,.7)]:
   with self.assertRaises(ValueError):gate.eligible(dict(AP50=pair[0],AP50_95=pair[1]))
if __name__=='__main__':unittest.main()
