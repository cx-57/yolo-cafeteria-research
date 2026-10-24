import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

RUN=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('receipt',RUN/'benchmark_receipt.py');receipt=importlib.util.module_from_spec(spec);spec.loader.exec_module(receipt)
spec=importlib.util.spec_from_file_location('shared',RUN.parent/'joint-metric-v30/policy.py');policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

class ReceiptTests(unittest.TestCase):
 def fixture(self,tmp):
  run=Path(tmp)/'run';run.mkdir();candidate=run/'chosen.pt';candidate.write_bytes(b'trained model');digest=sha(candidate);snapshot=run/f'verified-{digest}.pt';snapshot.write_bytes(candidate.read_bytes());audit=run/'dataset.json';audit.write_text(json.dumps(dict(fixed_source_fingerprint='source',validation_protocol='protocol')))
  manifest=dict(source_fingerprint='source',validation_protocol='protocol')
  report=dict(weights=str(snapshot),source_checkpoint=str(candidate),weights_sha=digest,role='monitor-selected',completed_epoch=9,AP50=.91,AP50_95=.65,precision=.9,recall=.9,imgsz=640,batch=4,nms_iou=.5,test_time_augmentation=False,device='mps',validation_images=246,bottle_objects=67,training_dataset_audit_sha=sha(audit),source_dataset_fingerprint='source',validation_protocol='protocol',ultralytics_version='8.4.175',independent_test=False)
  p=run/'phase9-monitor-selected.json';p.write_text(json.dumps(report));return run,p,candidate,manifest,report

 def test_valid_report_without_history_recovers_idempotently_without_inference(self):
  with tempfile.TemporaryDirectory() as tmp:
   run,p,candidate,manifest,report=self.fixture(tmp)
   self.assertEqual(receipt.reconcile(run,p,candidate,manifest,policy),report)
   self.assertEqual(policy.verified_records(run),[report])
   receipt.reconcile(run,p,candidate,manifest,policy)
   self.assertEqual(policy.verified_records(run),[report]);self.assertEqual(json.loads((run/'frontier.json').read_text())['members'],[report])

 def test_wrong_weights_protocol_audit_and_configuration_cannot_be_recovered(self):
  with tempfile.TemporaryDirectory() as tmp:
   run,p,candidate,manifest,report=self.fixture(tmp)
   for key,value in [('weights_sha','bad'),('validation_protocol','changed'),('training_dataset_audit_sha','bad'),('nms_iou',.7),('validation_images',36),('completed_epoch',1)]:
    p.write_text(json.dumps(dict(report,**{key:value})))
    with self.subTest(field=key),self.assertRaises((AssertionError,ValueError)):receipt.reconcile(run,p,candidate,manifest,policy)
   p.write_text(json.dumps(report));Path(report['weights']).write_bytes(b'changed snapshot')
   with self.assertRaises(AssertionError):receipt.reconcile(run,p,candidate,manifest,policy)
   self.assertFalse((run/'metrics-history.json').exists())

 def test_recovered_winner_promotes_only_without_metric_regression(self):
  with tempfile.TemporaryDirectory() as tmp:
   run,p,candidate,manifest,report=self.fixture(tmp);inc=Path(tmp)/'verified-incumbent';inc.mkdir();old=inc/'bottle.pt';old.write_bytes(b'old');proof=dict(AP50=.84,AP50_95=.62,weights_sha=sha(old));(inc/'verification.json').write_text(json.dumps(proof));runtime=Path(tmp)/'runtime/bottle.pt'
   self.assertFalse(receipt.promote(run,p,dict(report,AP50_95=.61),runtime));self.assertEqual(old.read_bytes(),b'old')
   self.assertTrue(receipt.promote(run,p,report,runtime));self.assertEqual(old.read_bytes(),candidate.read_bytes());self.assertEqual(runtime.read_bytes(),candidate.read_bytes());self.assertEqual(json.loads((inc/'verification.json').read_text())['weights_sha'],report['weights_sha'])
   self.assertFalse(receipt.promote(run,p,report,runtime))

if __name__=='__main__':unittest.main()
