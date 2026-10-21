"""Protect scoped labels, source provenance, and frozen capture assignments."""
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import focus_three_class as focus


class PlateWrapperTests(unittest.TestCase):
    def implementation(self):
        self.assertIsNotNone(importlib.util.find_spec('plate_wrapper'),
                             'Two-class extraction and verification are not implemented')
        import plate_wrapper
        return plate_wrapper

    def source(self, root):
        source = root / 'source'
        records = []
        both = ['0 0.3 0.3 0.2 0.2', '1 0.6 0.6 0.2 0.2', '2 0.8 0.8 0.1 0.1']
        for color, split, rows in [('red', 'train', both), ('green', 'train', [both[2]]),
                                    ('yellow', 'train', []), ('blue', 'val', both)]:
            name = f'fixture__{color}.jpg'
            image = source / 'images' / split / name
            label = source / 'labels' / split / f'fixture__{color}.txt'
            image.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (32, 32), color).save(image)
            label.write_text('\n'.join(rows) + ('\n' if rows else ''))
            records.append(dict(name=name, split=split, original_split=split,
                                group=f'fixture__{color}.jpg', sha=focus.digest(image),
                                pixels=focus.pixels(image), rows=list(rows), verified_absence=not rows,
                                review_status='verified_unchanged', accepted=True,
                                input_label_sha256=focus.digest(label), override_applied=True,
                                frozen_train=split == 'train', frozen_val=split == 'val'))
        manifest = dict(records=records, selected=4, counts=focus.audit(records, 4),
                        fingerprint=focus.fingerprint(records), scene_reviews=[],
                        validation_protocol=focus.fingerprint([records[-1]]))
        (source / 'manifest.json').write_text(json.dumps(manifest))
        (source / 'data.yaml').write_text(yaml.safe_dump(dict(path=str(source), train='images/train',
                                                            val='images/val', names=focus.NAMES)))
        return source

    def test_build_preserves_membership_boxes_and_source_bytes(self):
        module = self.implementation()
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(Path(directory).resolve())
            before = {str(p.relative_to(source)): focus.digest(p) for p in source.rglob('*') if p.is_file()}
            data = source.parent / 'two-class'
            module.build(source, data)
            result = module.verify(data)
            self.assertEqual(result['counts']['train']['images'], 3)
            self.assertEqual(result['counts']['train']['negatives'], 2)
            self.assertEqual(result['counts']['val']['objects'], dict(plate=1, wrapper=1))
            self.assertEqual((data / 'labels/train/fixture__red.txt').read_text().splitlines(),
                             ['0 0.3 0.3 0.2 0.2', '1 0.6 0.6 0.2 0.2'])
            self.assertEqual((data / 'labels/train/fixture__green.txt').read_text(), '')
            self.assertEqual(before, {str(p.relative_to(source)): focus.digest(p)
                                      for p in source.rglob('*') if p.is_file()})
            with self.assertRaises(ValueError):
                module.build(source, data)

    def test_verify_rejects_removed_target_box_even_with_rehashed_manifest(self):
        module = self.implementation()
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(Path(directory).resolve())
            data = source.parent / 'two-class'
            module.build(source, data)
            manifest = json.loads((data / 'manifest.json').read_text())
            record = next(r for r in manifest['records'] if r['name'] == 'fixture__red.jpg')
            record['rows'].pop()
            (data / 'labels/train/fixture__red.txt').write_text('\n'.join(record['rows']) + '\n')
            manifest['fingerprint'] = focus.fingerprint(manifest['records'])
            (data / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'projection|source|preserv'):
                module.verify(data)

    def test_unresolved_annotations_require_hash_bound_review(self):
        module = self.implementation()
        records = [dict(name='fixture__a.jpg', sha='image', rows=['2 0.5 0.5 0.2 0.2'],
                        verified_absence=False, review_status='needs_adjudication')]
        with self.assertRaisesRegex(ValueError, 'review|Unresolved'):
            module.project_records(records, [])
        self.assertEqual(records[0]['rows'], ['2 0.5 0.5 0.2 0.2'])

    def test_training_review_cannot_change_frozen_validation_labels(self):
        module = self.implementation()
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(Path(directory).resolve())
            records = json.loads((source / 'manifest.json').read_text())['records']
            record = records[-1]
            review = dict(name=record['name'], image_sha256=record['sha'],
                          input_rows_sha256=hashlib.sha256(json.dumps(record['rows'], sort_keys=True).encode()).hexdigest(),
                          status='corrected', rows=record['rows'][:1], reviewer='fixture-review',
                          reason='This would improperly remove frozen validation targets')
            path = source.parent / 'review.json'
            path.write_text(json.dumps([review]))
            with self.assertRaisesRegex(ValueError, 'Validation|validation'):
                module.project_records(records, [path])

    def test_two_class_audit_still_rejects_scene_leakage_and_unverified_negatives(self):
        self.implementation()
        rows = ['0 0.3 0.3 0.2 0.2', '1 0.6 0.6 0.2 0.2']
        records = [dict(name=n, split=s, group=n, sha=n, pixels=n, rows=list(rows),
                        verified_absence=False, scene_group='same-scene')
                   for n, s in [('a', 'train'), ('b', 'val')]]
        with self.assertRaisesRegex(ValueError, 'leakage'):
            focus.audit(records, 2, names=['plate', 'wrapper'])
        records[1]['scene_group'] = 'other-scene'
        records[0]['rows'] = []
        with self.assertRaisesRegex(ValueError, 'Unverified negative'):
            focus.audit(records, 2, names=['plate', 'wrapper'])

    def test_explicit_rebaseline_preserves_images_and_records_changed_validation(self):
        module = self.implementation()
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(Path(directory).resolve())
            r = json.loads((source / 'manifest.json').read_text())['records'][-1]
            review = dict(name=r['name'], image_sha256=r['sha'],
                          input_rows_sha256=hashlib.sha256(json.dumps(r['rows'], sort_keys=True).encode()).hexdigest(),
                          status='corrected', rows=r['rows'] + ['0 0.85 0.2 0.1 0.1'],
                          reviewer='fixture-review', reason='Additional visible plate', scope=['plate', 'wrapper'])
            path = source.parent / 'review.json'
            path.write_text(json.dumps([review]))
            data = source.parent / 'corrected'
            module.build(source, data, [path], rebaseline=True)
            result = module.verify(data)
            self.assertTrue(result['benchmark_revised'])
            self.assertEqual(result['selected'], 4)
            self.assertEqual(result['counts']['val']['objects'], dict(plate=2, wrapper=1))
            self.assertEqual(len(result['validation_label_changes']), 1)
            self.assertEqual(result['validation_label_changes'][0]['name'], 'fixture__blue.jpg')

    def test_quality_hold_blocks_verification_of_a_known_bad_dataset(self):
        module = self.implementation()
        with tempfile.TemporaryDirectory() as directory:
            source = self.source(Path(directory).resolve())
            data = source.parent / 'two-class'
            module.build(source, data)
            (data / 'quality_hold.json').write_text(json.dumps(dict(reason='Known false negative',
                                                                   replacement='new-reviewed-dataset')))
            with self.assertRaisesRegex(ValueError, 'quality|Quality'):
                module.verify(data)

    def test_rebaseline_report_cannot_mix_datasets_or_changed_weights(self):
        module = self.implementation()
        validate = getattr(module, 'validate_baseline_report', None)
        self.assertIsNotNone(validate, 'Corrected benchmarks need a checked baseline report')
        with tempfile.TemporaryDirectory() as directory:
            weight = Path(directory) / 'initial.pt'
            weight.write_bytes(b'initial-weights')
            manifest = dict(fingerprint='corrected-data', validation_protocol='corrected-validation',
                            counts={'val': {'objects': {'plate': 2, 'wrapper': 1}}})
            entry = dict(weights=str(weight), weights_sha256=focus.digest(weight),
                         metrics={'per_class': {'plate': {'objects': 2}, 'wrapper': {'objects': 1}}})
            report = dict(dataset_fingerprint='corrected-data', validation_protocol='corrected-validation',
                          candidates={'filtered_three_class': entry, 'two_class_epoch1': entry})
            validate(report, manifest)
            report['validation_protocol'] = 'old-validation'
            with self.assertRaisesRegex(ValueError, 'baseline|Baseline'):
                validate(report, manifest)
            report['validation_protocol'] = 'corrected-validation'
            weight.write_bytes(b'replaced-weights')
            with self.assertRaisesRegex(ValueError, 'weight|checkpoint'):
                validate(report, manifest)

    def test_filtered_baseline_must_have_zero_bottle_validation_support(self):
        module = self.implementation()
        metrics = SimpleNamespace(nt_per_class=[2, 1, 0], box=SimpleNamespace(
            ap_class_index=[0, 1], p=[.8, .9], r=[.7, .8], ap50=[.85, .9], ap=[.6, .7],
            mp=.85, mr=.75, map50=.875, map=.65))
        manifest = dict(counts={'val': {'objects': {'plate': 2, 'wrapper': 1}}})
        result = module.metrics_result(metrics, manifest, filtered_bottle=True)
        self.assertEqual(result['per_class']['plate']['AP50'], .85)
        metrics.nt_per_class = [2, 1, 1]
        with self.assertRaisesRegex(ValueError, 'support'):
            module.metrics_result(metrics, manifest, filtered_bottle=True)

    def test_warm_start_cannot_have_trained_on_new_validation_images(self):
        module = self.implementation()
        validate = getattr(module, 'validate_warm_split', None)
        self.assertIsNotNone(validate, 'Warm-start membership must be checked across label revisions')
        earlier = [dict(name='a', split='train', sha='a'), dict(name='b', split='val', sha='b')]
        revised = [dict(r) for r in earlier]
        validate(earlier, revised)
        revised[0]['split'] = 'val'
        with self.assertRaisesRegex(ValueError, 'split|membership'):
            validate(earlier, revised)

    def test_lr_comparison_keeps_optimizer_state_and_source_unchanged(self):
        module = self.implementation()
        scale = getattr(module, 'scaled_continuation', None)
        self.assertIsNotNone(scale, 'Matched learning-rate branches need an optimizer-preserving clone')
        checkpoint = dict(epoch=4, optimizer=dict(state={'momentum': [1, 2]},
                          param_groups=[dict(lr=.00095, initial_lr=.001)]),
                          train_args=dict(epochs=80, lr0=.001, data='audited/data.yaml'))
        clone = scale(checkpoint, Path('/tmp/new-output'), Path('/tmp/data'), .3)
        self.assertEqual(clone['epoch'], 4)
        self.assertEqual(clone['optimizer']['state'], {'momentum': [1, 2]})
        self.assertAlmostEqual(clone['train_args']['lr0'], .0003)
        self.assertAlmostEqual(clone['optimizer']['param_groups'][0]['lr'], .000285)
        self.assertAlmostEqual(clone['optimizer']['param_groups'][0]['initial_lr'], .0003)
        self.assertEqual(checkpoint['optimizer']['param_groups'][0]['lr'], .00095)
        self.assertEqual(checkpoint['train_args']['lr0'], .001)
        self.assertEqual(clone['train_args']['epochs'], 80)
        for factor in [0, -1, 2, float('nan')]:
            with self.assertRaises(ValueError):
                scale(checkpoint, Path('/tmp/new-output'), Path('/tmp/data'), factor)

    def test_lr_branch_writes_new_provenance_and_retains_verified_best(self):
        module = self.implementation()
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / 'original/train/weights/best_min_ap50.pt'
            source.parent.mkdir(parents=True)
            data = root / 'data'
            checkpoint = dict(epoch=4, optimizer=dict(state={}, param_groups=[dict(lr=.00095, initial_lr=.001)]),
                              train_args=dict(epochs=80, lr0=.001, data=str(data / 'data.yaml')))
            torch.save(checkpoint, source)
            prior = dict(dataset_fingerprint='fixed', validation_protocol='fixed-val', epochs=80, target_AP50=.9,
                         source_weights='initial.pt', source_weights_sha256='old', initialization='old',
                         baseline_metrics={'old': True})
            (root / 'original/experiment.json').write_text(json.dumps(prior))
            (root / 'original/minimum_class_selection.json').write_text(json.dumps(dict(
                weights=str(source), weights_sha256=focus.digest(source), epoch=5, minimum_AP50=.85)))
            (root / 'original/train/results.csv').write_text('epoch,time\n1,1\n2,2\n3,3\n4,4\n5,5\n')
            args = SimpleNamespace(continue_from=source, model=root / 'branch', data=data, lr_scale=.3,
                                   target=.9, resume=False)
            # Real evaluation invokes MPS inference; this test covers branch files and optimizer state.
            with patch('plate_wrapper.evaluate', return_value={'per_class': {'plate': {'AP50': .85}}}):
                module.prepare_lr_branch(args, dict(fingerprint='fixed', validation_protocol='fixed-val'))
            result = json.loads((args.model / 'experiment.json').read_text())
            self.assertEqual(result['source_weights'], str(source))
            self.assertEqual(result['source_completed_epochs'], 5)
            self.assertEqual(result['lr_scale'], .3)
            self.assertEqual(result['baseline_metrics']['per_class']['plate']['AP50'], .85)
            self.assertEqual(focus.digest(source), focus.digest(args.model / 'train/weights/best_min_ap50.pt'))
            self.assertEqual(prior, json.loads((root / 'original/experiment.json').read_text()))

    def test_resume_cannot_silently_ignore_requested_lr_change(self):
        module = self.implementation()
        with patch.object(sys, 'argv', ['plate_wrapper.py', 'train', '--resume', '--lr-scale', '.3']):
            with patch('plate_wrapper.train'):
                with self.assertRaises(SystemExit) as raised:
                    module.main()
                self.assertEqual(raised.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
