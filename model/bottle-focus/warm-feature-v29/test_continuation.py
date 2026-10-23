"""Continuation must trust a matching saved-model evaluation, never native training AP."""
from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import subprocess
import tempfile
import unittest

RUN=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('bottle_continuation',RUN/'continue_training.py')
driver=importlib.util.module_from_spec(spec)
if Path(spec.origin).exists():spec.loader.exec_module(driver)

class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(driver,'verified_phase'),'saved-model verification gate missing')
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.run=Path(self.tmp.name)/'warm-feature-v29';self.run.mkdir()
        weights=self.run/'best_trained_bottle.pt';weights.write_bytes(b'fixed local checkpoint bytes')
        digest=hashlib.sha256(weights.read_bytes()).hexdigest()
        self.save('dataset.json',dict(fixed_source_fingerprint='frozen-source',validation_protocol='frozen-protocol'))
        self.save('progress.json',dict(epoch=3,AP50=.99))
        self.save('trained-selection.json',dict(epoch=2,weights_sha=digest))
        self.report=dict(weights=str(weights),weights_sha=digest,selected_epoch=2,AP50=.8499,imgsz=640,batch=4,nms_iou=.5,test_time_augmentation=False,validation_images=246,bottle_objects=67,source_dataset_fingerprint='frozen-source',validation_protocol='frozen-protocol')
        self.report['training_dataset_audit_sha']=hashlib.sha256((self.run/'dataset.json').read_bytes()).hexdigest()
        self.save('independent-trained-ordinary640-iou0.5.json',self.report)

    def save(self,name,data):
        (self.run/name).write_text(json.dumps(data))

    def test_native_crossing_does_not_override_saved_model_score(self):
        epoch,report=driver.verified_phase(self.run)
        self.assertEqual(epoch,3)
        self.assertEqual(report['AP50'],.8499)

    def test_changed_model_or_evaluation_protocol_is_rejected(self):
        for key,value in [('weights_sha','0'*64),('nms_iou',.7),('bottle_objects',68),('validation_protocol','changed'),('AP50',float('nan')),('selected_epoch',1),('training_dataset_audit_sha','0'*64)]:
            with self.subTest(key=key):
                changed=dict(self.report);changed[key]=value
                self.save('independent-trained-ordinary640-iou0.5.json',changed)
                with self.assertRaises((AssertionError,ValueError)):driver.verified_phase(self.run)
        self.save('independent-trained-ordinary640-iou0.5.json',self.report)
        (self.run/'best_trained_bottle.pt').write_bytes(b'changed checkpoint')
        with self.assertRaises((AssertionError,ValueError)):driver.verified_phase(self.run)

    def test_restart_archives_verified_target_without_launching_training(self):
        shutil.copytree(self.run,self.run.parent/'reviewed-expansion-v28')
        report=dict(self.report,AP50=.85)
        self.save('independent-trained-ordinary640-iou0.5.json',report)
        original=driver.RUN;driver.RUN=self.run
        self.addCleanup(setattr,driver,'RUN',original)
        try:driver.main()
        except subprocess.CalledProcessError:self.fail('Restart launched training despite an already verified target')
        state=json.loads((self.run/'continuation.json').read_text())
        self.assertEqual(state['status'],'target_verified')
        self.assertEqual(state['phases'][-1]['AP50'],.85)
        self.assertTrue(Path(state['phases'][-1]['snapshot']).exists())

    def test_interrupted_phase_requires_fresh_verification_before_more_training(self):
        shutil.copytree(self.run,self.run.parent/'reviewed-expansion-v28')
        (self.run/'independent-trained-ordinary640-iou0.5.json').unlink()
        original=driver.RUN;driver.RUN=self.run
        self.addCleanup(setattr,driver,'RUN',original)
        with self.assertRaisesRegex(RuntimeError,'fresh verification'):driver.main()
        self.assertEqual(json.loads((self.run/'continuation.json').read_text())['status'],'verification_required')
        self.assertFalse(list(self.run.glob('phase-from-*.log')))

if __name__=='__main__':unittest.main()
