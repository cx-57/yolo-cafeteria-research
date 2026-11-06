import copy
from pathlib import Path
import sys
import tempfile
import unittest
import importlib.util
import json

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    import v5_trials as t
except ImportError:
    t = None


class TrialTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(t, 'unified trial runner is missing')

    def metrics(self, ap=.5, recall=.5):
        return {n: dict(AP50=.7, AP50_95=ap, recall_at_precision_80=recall,
                        threshold_at_precision_80=.5) for n in
                ('food', 'plate', 'utensil', 'hinged_container', 'drink_carton', 'wrapper')}

    def record(self, trial, seed=57):
        return dict(trial=trial, seed=seed, status='complete', smoke=False, dataset_sha256='same',
                    completed_epochs=40, planned_epochs=40, per_class=self.metrics(),
                    checkpoint_names=['food', 'plate', 'utensil', 'hinged_container', 'drink_carton', 'wrapper'],
                    test_evaluated=False)

    def test_recipes_use_real_budgets_and_fresh_pretraining(self):
        base = t.trial_recipe('baseline', 57)
        self.assertEqual(base['epochs'], 40)
        self.assertEqual(base['pretrained'], 'yolo11s.pt')
        self.assertEqual(base['patience'], 40)
        self.assertEqual(t.trial_recipe('U2', 57)['imgsz'], 1024)
        self.assertEqual(t.trial_recipe('U5', 57)['pretrained'], 'yolo11m.pt')
        with self.assertRaisesRegex(ValueError, 'U1.*U5|prerequisite'):
            t.trial_recipe('U6', 57)

    def test_user_can_request_ten_epochs_without_unlocking_specialists(self):
        self.assertTrue(hasattr(t, 'with_epoch_budget'), 'user epoch override is missing')
        original = t.trial_recipe('baseline', 57)
        recipe = t.with_epoch_budget(original, 10)
        self.assertEqual(recipe['epochs'], 10)
        self.assertEqual(recipe['patience'], 10)
        self.assertEqual(original['epochs'], 40)
        for count in (0, -1, True, 1.5):
            with self.assertRaises(ValueError): t.with_epoch_budget(original, count)
        record = self.record('baseline')
        record.update(planned_epochs=10, completed_epochs=10, recipe=recipe)
        self.assertFalse(t.valid_completed(record, 'same'))

    def test_u6_extends_convergence_when_no_variant_helps(self):
        runs = [self.record(k) for k in t.TRIALS[:-1]]
        recipe = t.trial_recipe('U6', 57, runs)
        self.assertEqual(recipe['epochs'], 80)
        self.assertEqual(recipe['pretrained'], 'yolo11s.pt')

    def test_balancing_is_capped_and_never_uses_holdouts(self):
        records = [dict(split='train', image_path='images/train/a.png', counts={'2': 1}),
                   dict(split='train', image_path='images/train/b.png', counts={'0': 1}),
                   dict(split='val', image_path='images/val/c.png', counts={'2': 9}),
                   dict(split='test', image_path='images/test/d.png', counts={'2': 9})]
        paths = t.balanced_training_paths(records, ['utensil'])
        self.assertEqual(paths.count('images/train/a.png'), 3)
        self.assertEqual(paths.count('images/train/b.png'), 1)
        self.assertFalse(any('/val/' in p or '/test/' in p for p in paths))

    def test_failed_smoke_interrupted_wrong_split_and_nonfinite_not_completed(self):
        for patch in (dict(status='failed'), dict(smoke=True), dict(completed_epochs=11),
                      dict(dataset_sha256='other'), dict(test_evaluated=True),
                      dict(checkpoint_names=['bottle']), dict(per_class=self.metrics(float('nan')))):
            with self.subTest(patch=patch):
                r = self.record('baseline')
                r.update(patch)
                self.assertFalse(t.valid_completed(r, 'same'))

    def test_gate_stays_closed_before_all_trials_and_repeats(self):
        runs = [self.record(k) for k in t.TRIALS]
        report = t.summarize_stage(runs, 'same')
        self.assertFalse(report['specialists_allowed'])
        self.assertEqual(report['status'], 'confirmation_runs_required')
        self.assertEqual(report['selected_unified_trial'], 'baseline')
        runs += [self.record('baseline', seed) for seed in (58, 59)]
        report = t.summarize_stage(runs, 'same')
        self.assertFalse(report['specialists_allowed'])
        self.assertEqual(report['status'], 'uncertainty_analysis_required')

    def test_gate_rejects_missing_duplicate_and_tampered_runs(self):
        runs = [self.record(k) for k in t.TRIALS[:-1]]
        report = t.summarize_stage(runs, 'same')
        self.assertEqual(report['missing_trials'], ['U6'])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            t.summarize_stage(runs + [runs[0]], 'same')
        tampered = copy.deepcopy(runs)
        tampered[1]['per_class']['utensil']['AP50_95'] = 2
        self.assertIn('U1', t.summarize_stage(tampered, 'same')['missing_trials'])

    def test_selection_rejects_regressions_in_strong_classes(self):
        runs = [self.record(k) for k in t.TRIALS]
        runs[1]['per_class']['utensil']['AP50_95'] = .8
        runs[1]['per_class']['food']['AP50_95'] = .4
        self.assertEqual(t.summarize_stage(runs, 'same')['selected_unified_trial'], 'baseline')

    def test_operating_point_reports_unattainable_precision(self):
        self.assertEqual(t.operating_point([0, .5, 1], [.6, .7, 1], [.9, .4, 0]),
                         dict(recall_at_precision_80=0.0, threshold_at_precision_80=None))
        self.assertEqual(t.operating_point([0, .5, 1], [.5, .9, 1], [.9, .6, 0]),
                         dict(recall_at_precision_80=.6, threshold_at_precision_80=.5))

    def test_artifact_receipt_detects_modified_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'best.pt').write_bytes(b'weights')
            receipt = dict(artifacts={'best.pt': t.sha(root / 'best.pt')})
            t.verify_artifacts(root, receipt)
            (root / 'best.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Changed'):
                t.verify_artifacts(root, receipt)

    def test_resume_never_retrains_finished_or_stripped_checkpoints(self):
        self.assertTrue(hasattr(t, 'resume_action'), 'safe phase-aware resume is missing')
        self.assertEqual(t.resume_action(dict(phase='evaluation', completed_epochs=40),
                                         dict(epoch=-1, optimizer=None), 40), 'evaluate')
        self.assertEqual(t.resume_action(dict(phase='training'),
                                         dict(epoch=10, optimizer={'state': {}}), 40), 'train')
        for ckpt in (dict(epoch=-1, optimizer=None), dict(epoch=39, optimizer={'state': {}}),
                     dict(epoch=10, optimizer=None)):
            with self.assertRaisesRegex(ValueError, 'resumable'):
                t.resume_action(dict(phase='training'), ckpt, 40)

    def test_cached_labels_are_removed_before_frozen_labels_are_loaded(self):
        self.assertTrue(hasattr(t, 'clear_label_caches'), 'label cache invalidation is missing')
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'labels').mkdir()
            cache = root / 'labels/train.cache'
            cache.write_bytes(b'untrusted cached labels')
            unrelated = root / 'keep.txt'
            unrelated.write_text('keep')
            t.clear_label_caches(root)
            self.assertFalse(cache.exists())
            self.assertTrue(unrelated.is_file())

    def test_resume_rejects_changed_runtime(self):
        self.assertTrue(hasattr(t, 'verify_runtime'), 'runtime comparison is missing')
        old = dict(python='3.12.3', torch='2.6.0', ultralytics='8.4.175', platform='a')
        new = dict(old, ultralytics='8.4.203')
        with self.assertRaisesRegex(ValueError, 'environment'):
            t.verify_runtime(old, new)
        t.verify_runtime(old, dict(old, platform='another GPU host'))

    def test_disk_preflight_fails_before_creating_run_output(self):
        self.assertTrue(hasattr(t, 'ensure_disk_space'), 'disk-space preflight is missing')
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / 'new-run'
            with self.assertRaisesRegex(ValueError, 'disk space'):
                t.ensure_disk_space(output, minimum_bytes=10**20)
            self.assertFalse(output.exists())
            t.ensure_disk_space(output, minimum_bytes=0)

    @unittest.skipUnless(importlib.util.find_spec('ultralytics'), 'Ultralytics is not installed')
    def test_validation_adapter_saves_per_image_stats_without_training_epochs(self):
        import torch
        import ultralytics
        from ultralytics import YOLO
        from v5_dataset import NAMES
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'images/val').mkdir(parents=True)
            (root / 'labels/val').mkdir(parents=True)
            Image.new('RGB', (64, 64), 'gray').save(root / 'images/val/00001.png')
            (root / 'labels/val/00001.txt').write_text(''.join(
                f'{c} .5 .5 .2 .2\n' for c in range(6)))
            data = root / 'data.yaml'
            data.write_text(yaml.safe_dump(dict(path=str(root), train='images/val', val='images/val',
                                               nc=6, names=NAMES)))
            cfg = yaml.safe_load((Path(ultralytics.__file__).parent / 'cfg/models/11/yolo11.yaml').read_text())
            cfg.update(nc=6, scale='n')
            model_file = root / 'yolo11n.yaml'
            model_file.write_text(yaml.safe_dump(cfg))
            output = root / 'output'
            output.mkdir()
            model = YOLO(str(model_file), task='detect')
            model.model.names = dict(enumerate(NAMES))
            metrics, speed = t.evaluate(model, data, output,
                                        {'imgsz': 64}, 'cpu', 1)
            stats = json.loads((output / 'validation_stats.json').read_text())
            self.assertEqual([r['name'] for r in stats], ['00001.png'])
            self.assertEqual(stats[0]['target_cls'], [0., 1., 2., 3., 4., 5.])
            self.assertEqual(set(metrics), set(NAMES))
            self.assertEqual(metrics['utensil']['recall_at_precision_80'], 0)
            self.assertTrue((output / 'curves.json').is_file())


if __name__ == '__main__': unittest.main()
