"""Exercise actual finalizer: a high monitor score cannot trigger an early benchmark."""
from pathlib import Path
import importlib.util
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ML=importlib.util.find_spec('torch') is not None and importlib.util.find_spec('ultralytics') is not None

class LifecycleTests(unittest.TestCase):
 @unittest.skipUnless(ML,'requires ML dependencies; covered in ML integration')
 def test_actual_finalizer_below_monitor_gate_skips_benchmark_and_does_not_misattribute_reference(self):
  import hashlib
  path=Path(__file__).with_name('finish.py');spec=importlib.util.spec_from_file_location('v36_gate_finish',path);finish=importlib.util.module_from_spec(spec);spec.loader.exec_module(finish)
  with tempfile.TemporaryDirectory() as directory:
   run=Path(directory)/'run';run.mkdir();finish.RUN=run;weights=Path(directory)/'weights';weights.mkdir()
   for name,raw in [('initial.pt',b'initial'),('best-monitor-ap50.pt',b'trained'),('best-monitor-localization.pt',b'trained'),('last_resume.pt',b'trained')]:
    (weights/name).write_bytes(raw)
   sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
   store=SimpleNamespace(load=lambda p:json.loads(p.read_text()),save=lambda p,v:Path(p).write_text(json.dumps(v)),sha=sha)
   for role in ['ap50','localization']:store.save(run/f'native-{role}-selection.json',dict(epoch=7))
   store.save(run/'monitor-progress.json',dict(epoch=9))
   reference=Path(directory)/'joint-metric-v30';reference.mkdir()
   store.save(reference/'phase0-initial.json',dict(weights_sha=sha(weights/'initial.pt'),source_dataset_fingerprint='fixed',validation_protocol='protocol',AP50=.8407,AP50_95=.6206))
   common=SimpleNamespace(s=store,MANIFEST=dict(source_fingerprint='fixed',validation_protocol='protocol'))
   audit=dict(monitor_fingerprint='monitor');dataset=SimpleNamespace(verify=lambda:audit)
   store.save(run/'dataset.json',audit)
   def measured(run,path,role,epoch,common,audit,weights_dir):
    return dict(role=role,selected_epoch=epoch,AP50=.62 if role=='initial' else .80,AP50_95=.51 if role=='initial' else .63,metric_source='continuation_monitor',monitor_fingerprint='monitor',weights=str(path),weights_sha=sha(path))
   with patch.object(finish.monitor_eval,'evaluate',side_effect=measured),patch.object(finish.benchmark,'evaluate',side_effect=AssertionError('benchmark invoked')) as inference:
    result=finish.finish(common,dataset,weights)
   inference.assert_not_called();self.assertFalse(result['new_benchmark_measurement']);self.assertIsNone(result['AP50'])
   self.assertEqual(result['benchmark_reference_AP50'],.8407);self.assertFalse((run/'benchmark-attempt.json').exists())

 @unittest.skipUnless(ML,'requires ML dependencies; covered in ML integration')
 def test_actual_finalizer_rejects_early_monitor_success_without_benchmarking(self):
  path=Path(__file__).with_name('finish.py');spec=importlib.util.spec_from_file_location('v34_test_finish',path);finish=importlib.util.module_from_spec(spec);spec.loader.exec_module(finish)
  with tempfile.TemporaryDirectory() as directory:
   finish.RUN=Path(directory);(finish.RUN/'monitor-progress.json').write_text(json.dumps(dict(epoch=1,AP50=.99,AP50_95=.90,metric_source='continuation_monitor')))
   common=SimpleNamespace(s=SimpleNamespace(load=lambda p:json.loads(p.read_text())))
   dataset=SimpleNamespace(verify=lambda:dict(monitor_fingerprint='monitor'))
   with patch.object(finish.benchmark,'evaluate',side_effect=AssertionError('benchmark invoked')) as call:
    with self.assertRaises(AssertionError):finish.finish(common,dataset,Path(directory))
    call.assert_not_called()
   self.assertFalse((finish.RUN/'benchmark-attempt.json').exists())

 @unittest.skipUnless(ML,'requires ML dependencies; covered in ML integration')
 def test_actual_finalizer_recovers_report_before_history_without_inference(self):
  import contextlib,hashlib,io
  root=Path(__file__).parent
  spec=importlib.util.spec_from_file_location('v34_receipt_fixture',root/'test_receipt.py');fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
  spec=importlib.util.spec_from_file_location('v34_recovery_finish',root/'finish.py');finish=importlib.util.module_from_spec(spec);spec.loader.exec_module(finish)
  with tempfile.TemporaryDirectory() as directory:
   run,report_path,candidate,manifest,report=fixture.ReceiptTests().fixture(directory);finish.RUN=run;weights=Path(directory)/'weights';weights.mkdir()
   (weights/'initial.pt').write_bytes(b'initial')
   for name in ['best-monitor-ap50.pt','best-monitor-localization.pt','last_resume.pt']:(weights/name).write_bytes(candidate.read_bytes())
   for role in ['ap50','localization']:(run/f'native-{role}-selection.json').write_text(json.dumps(dict(epoch=6)))
   (run/'monitor-progress.json').write_text(json.dumps(dict(epoch=9)))
   incumbent=Path(directory)/'verified-incumbent';incumbent.mkdir();(incumbent/'bottle.pt').write_bytes(b'old');(incumbent/'verification.json').write_text(json.dumps(dict(AP50=.84,AP50_95=.62,weights_sha=fixture.sha(incumbent/'bottle.pt'))))
   store=SimpleNamespace(load=lambda p:json.loads(p.read_text()),save=lambda p,v:Path(p).write_text(json.dumps(v)),sha=fixture.sha)
   common=SimpleNamespace(s=store,MANIFEST=manifest,DATA=Path(directory)/'runtime/data/capture')
   audit=dict(monitor_fingerprint='monitor',monitor_images=36,monitor_bottle_boxes=17)
   dataset=SimpleNamespace(verify=lambda:audit)
   def measured(run,path,role,epoch,common,audit,weights_dir):
    return dict(role=role,selected_epoch=epoch,AP50=.84 if role=='initial' else .92,AP50_95=.62 if role=='initial' else .70,metric_source='continuation_monitor',monitor_fingerprint='monitor',weights=str(path if role=='initial' else candidate),weights_sha=fixture.sha(path))
   with patch.object(finish.monitor_eval,'evaluate',side_effect=measured),patch.object(finish.benchmark,'evaluate',side_effect=AssertionError('must reuse saved report')) as inference,contextlib.redirect_stdout(io.StringIO()):
    result=finish.finish(common,dataset,weights)
   inference.assert_not_called();self.assertEqual(result['status'],'joint_target_verified')
   self.assertEqual(finish.shared.verified_records(run),[report]);self.assertEqual((incumbent/'bottle.pt').read_bytes(),candidate.read_bytes())

if __name__=='__main__':unittest.main()
