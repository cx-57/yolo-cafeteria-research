import unittest
import tempfile
from pathlib import Path
from protocol import eligible,bind_attempt,receipt
class ProtocolTests(unittest.TestCase):
 def test_fresh_and_old_sources_must_all_pass(self):
  baseline=[.99,.9,.6,.5,.4,.3,.8,.6]
  self.assertTrue(eligible([.99,.9,.7,.6,.5,.4,.9,.7],baseline))
  self.assertFalse(eligible([.99,.9,.7,.6,.5,.4,.79,.7],baseline))
  self.assertFalse(eligible([.99,.89,.7,.6,.5,.4,.9,.7],baseline))
  self.assertFalse(eligible(baseline,baseline))
 def test_invalid_metrics_fail(self):
  for values in [[.9]*6,[float('nan')]*8,[.1,.2]*4]:
   with self.assertRaises(AssertionError):eligible(values,[.9]*8)
 def test_attempt_binds_settings_and_weights(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'attempt.json';bind_attempt(path,dict(weights='same',imgsz=896));bind_attempt(path,dict(weights='same',imgsz=896))
   with self.assertRaises(AssertionError):bind_attempt(path,dict(weights='same',imgsz=960))
 def test_receipt_rejects_wrong_resolution(self):
  report=dict(imgsz=640,weights_sha='same',AP50=.8,AP50_95=.6)
  self.assertEqual(receipt(report,dict(imgsz=640,weights_sha='same')),report)
  with self.assertRaises(AssertionError):receipt(report,dict(imgsz=896,weights_sha='same'))
if __name__=='__main__':unittest.main()
