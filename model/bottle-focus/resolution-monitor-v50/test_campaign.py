import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import campaign

class CampaignTests(unittest.TestCase):
 def execute(self,regress):
  saved={};calls=[]
  def evaluate(cohort,imgsz):
   name=cohort['name'];calls.append((name,imgsz))
   if name=='original-benchmark':return dict(AP50=.91,AP50_95=.66)
   a,b=(.6,.5) if imgsz==640 else (.7,.6)
   if name=='fresh' and imgsz==896 and regress:a=.59
   return dict(AP50=a,AP50_95=min(a,b))
  with tempfile.TemporaryDirectory() as tmp,patch.object(campaign,'RUN',Path(tmp)),patch.object(campaign,'identity',return_value={}),patch.object(campaign,'verify_monitor',return_value={}),patch.object(campaign,'cohorts',return_value=[dict(name=n) for n in ['retention','oi','litter','fresh']]),patch.object(campaign,'load',return_value={}),patch.object(campaign,'sha',return_value='sha'),patch.object(campaign,'save',side_effect=lambda p,r:saved.update({p.name:r})),patch.object(campaign.protocol,'bind_attempt'),patch.object(campaign,'frozen',return_value=dict(manifest_sha='manifest',source_dataset_fingerprint='fp',validation_protocol='old-protocol')),patch.object(campaign,'evaluate',side_effect=evaluate),patch.object(campaign,'YOLO',side_effect=AssertionError('No real ML in lifecycle fixture')):
   campaign.main()
  return saved,calls
 def test_failed_fresh_monitor_gate_skips_main_benchmark(self):
  saved,calls=self.execute(True)
  self.assertFalse(saved['campaign-result.json']['original_benchmark_used'])
  self.assertEqual(len(calls),8)
 def test_single_eligible_setting_uses_only_one_main_probe(self):
  saved,calls=self.execute(False)
  self.assertEqual(calls.count(('original-benchmark',896)),1)
  self.assertEqual(len(calls),9)
  self.assertTrue(saved['campaign-result.json']['move_on_eligible'])
  self.assertTrue(saved['campaign-result.json']['inference_protocol_changed'])
if __name__=='__main__':unittest.main()
