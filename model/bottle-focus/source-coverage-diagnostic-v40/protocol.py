"""Prevent old/source-different accuracy receipts from becoming diagnostic scores."""
import math
def validate_report(report,model_sha,dataset_sha,cohort_fingerprint,bottle_objects):
 expected=dict(weights_sha=model_sha,dataset_sha=dataset_sha,cohort_fingerprint=cohort_fingerprint,
               metric_source='fresh_litter_diagnostic',photos=32,bottle_objects=bottle_objects,imgsz=640,batch=4,nms_iou=.5,
               test_time_augmentation=False,original_benchmark_used=False,independent_NHS_test=False)
 assert all(report.get(k)==v for k,v in expected.items()),'Score belongs to different data/model/protocol'
 a,b=report['AP50'],report['AP50_95']
 if not all(math.isfinite(v) for v in (a,b)) or not 0<=b<=a<=1:raise ValueError('Invalid diagnostic metrics')
 return report
