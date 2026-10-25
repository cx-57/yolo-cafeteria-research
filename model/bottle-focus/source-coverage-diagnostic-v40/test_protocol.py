from pathlib import Path
import importlib.util
import unittest

spec=importlib.util.spec_from_file_location('v40_protocol',Path(__file__).with_name('protocol.py'));protocol=importlib.util.module_from_spec(spec);spec.loader.exec_module(protocol)
class ProtocolTests(unittest.TestCase):
 def report(self):return dict(weights_sha='model',dataset_sha='data',cohort_fingerprint='cohort',metric_source='fresh_litter_diagnostic',photos=32,bottle_objects=18,imgsz=640,batch=4,nms_iou=.5,test_time_augmentation=False,original_benchmark_used=False,independent_NHS_test=False,AP50=.8,AP50_95=.6)
 def test_saved_score_must_belong_to_exact_model_data_and_cohort(self):
  row=self.report();protocol.validate_report(row,'model','data','cohort',18)
  for key,value in [('weights_sha','other'),('dataset_sha','other'),('cohort_fingerprint','other'),('original_benchmark_used',True),('metric_source','fixed_benchmark'),('bottle_objects',67),('AP50',float('nan'))]:
   changed=dict(row);changed[key]=value
   with self.assertRaises((AssertionError,ValueError)):protocol.validate_report(changed,'model','data','cohort',18)
 def test_invalid_localization_pair_rejected(self):
  with self.assertRaises(ValueError):protocol.validate_report(dict(self.report(),AP50_95=.9),'model','data','cohort',18)
if __name__=='__main__':unittest.main()
