"""Joint success and tradeoffs must survive AP50-first selection pressure."""
from pathlib import Path
import importlib.util
import hashlib
import json
import tempfile
import unittest

RUN=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('joint_policy',RUN/'policy.py');policy=importlib.util.module_from_spec(spec)
if Path(spec.origin).exists():spec.loader.exec_module(policy)

class PolicyTests(unittest.TestCase):
    def setUp(self):self.assertTrue(hasattr(policy,'target_met'),'joint metric policy missing')

    def test_both_metrics_must_cross_without_rounding(self):
        for ap50,ap95,want in [(.90,.64,True),(.91,.70,True),(.89999,.70,False),(.94,.63999,False),(.80676,.64054,False)]:
            with self.subTest(ap50=ap50,ap95=ap95):self.assertEqual(policy.target_met(dict(AP50=ap50,AP50_95=ap95)),want)
        with self.assertRaises(ValueError):policy.target_met(dict(AP50=.95,AP50_95=float('nan')))

    def test_localization_tradeoff_survives_frontier_selection(self):
        rows=[dict(id='ap50',AP50=.8407837,AP50_95=.6090194),dict(id='balanced',AP50=.8407107,AP50_95=.6206057),dict(id='dominated',AP50=.83,AP50_95=.60)]
        self.assertEqual({r['id'] for r in policy.frontier(rows)},{'ap50','balanced'})
        winner=dict(id='joint',AP50=.90,AP50_95=.64)
        self.assertEqual([r['id'] for r in policy.frontier(rows+[winner])],['joint'])

    def test_saved_results_are_bound_to_weights_and_protocol(self):
        self.assertTrue(hasattr(policy,'verified_records'),'saved-result verification missing')
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);weights=run/'saved.pt';weights.write_bytes(b'immutable model')
            audit=dict(fixed_source_fingerprint='source',validation_protocol='protocol')
            (run/'dataset.json').write_text(json.dumps(audit))
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            report=dict(AP50=.90,AP50_95=.64,weights=str(weights),weights_sha=sha(weights),training_dataset_audit_sha=sha(run/'dataset.json'),source_dataset_fingerprint='source',validation_protocol='protocol',validation_images=246,bottle_objects=67,imgsz=640,batch=4,nms_iou=.5,test_time_augmentation=False)
            p=run/'metrics-history.json';p.write_text(json.dumps([report]))
            self.assertTrue(policy.target_met(policy.verified_records(run)[0]))
            for key,value in [('nms_iou',.7),('validation_protocol','other'),('training_dataset_audit_sha','0'*64)]:
                p.write_text(json.dumps([dict(report,**{key:value})]))
                with self.assertRaises((AssertionError,ValueError)):policy.verified_records(run)
            p.write_text(json.dumps([report]));weights.write_bytes(b'changed model')
            with self.assertRaises((AssertionError,ValueError)):policy.verified_records(run)

    def test_checkpoint_selection_keeps_localization_and_joint_winners(self):
        self.assertTrue(hasattr(policy,'save_roles'),'joint checkpoint selection missing')
        prior=dict(ap50=dict(AP50=.91,AP50_95=.63),localization=dict(AP50=.89,AP50_95=.65))
        self.assertEqual(policy.save_roles(dict(AP50=.90,AP50_95=.64),prior),{'joint'})
        self.assertEqual(policy.save_roles(dict(AP50=.88,AP50_95=.66),prior),{'localization'})
        self.assertEqual(policy.save_roles(dict(AP50=.92,AP50_95=.62),prior),{'ap50'})

    def test_plateau_stops_honestly_and_joint_success_takes_precedence(self):
        self.assertTrue(hasattr(policy,'phase_outcome'),'bounded phase policy missing')
        old=dict(reference=[.84,.62],stale_phases=2)
        stable=[dict(AP50=.84,AP50_95=.62)]
        self.assertEqual(policy.phase_outcome(old,stable,9)['status'],'screen_plateau_below_target')
        improved=policy.phase_outcome(old,[dict(AP50=.843,AP50_95=.62)],9)
        self.assertEqual((improved['status'],improved['stale_phases']),('running',0))
        self.assertEqual(policy.phase_outcome({},[dict(AP50=.91,AP50_95=.63999)],40)['status'],'schedule_complete_below_target')
        self.assertEqual(policy.phase_outcome(old,[dict(AP50=.90,AP50_95=.64)],40)['status'],'target_verified')

    def test_restart_does_not_bypass_a_stopped_screen(self):
        spec=importlib.util.spec_from_file_location('joint_driver',RUN/'continue_training.py');driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
        with tempfile.TemporaryDirectory() as directory:
            driver.RUN=Path(directory)
            state=dict(target_AP50=.90,minimum_AP50_95=.64,status='screen_plateau_below_target',stale_phases=3,phases=[])
            (driver.RUN/'continuation.json').write_text(json.dumps(state))
            try:driver.main()
            except FileNotFoundError:self.fail('Restart bypassed the plateau stop and attempted another phase')
            self.assertEqual(json.loads((driver.RUN/'continuation.json').read_text()),state)
            self.assertFalse(list(driver.RUN.glob('phase-*.log')))

if __name__=='__main__':unittest.main()
