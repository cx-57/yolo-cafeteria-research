import copy
from pathlib import Path
import sys
import tempfile
import json
import importlib.util
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    import v5_specialists as s
except ImportError:
    s = None


class SpecialistTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(s, 'V5 specialist implementation is missing')

    def test_labels_remap_owned_ids_and_retain_empty_negatives(self):
        text = '0 .5 .5 .2 .2\n2 .4 .4 .1 .1\n4 .6 .6 .2 .2\n'
        self.assertEqual(s.specialist_labels(text, [2]), '0 .4 .4 .1 .1\n')
        self.assertEqual(s.specialist_labels(text, [0, 3, 4]), '0 .5 .5 .2 .2\n2 .6 .6 .2 .2\n')
        self.assertEqual(s.specialist_labels('0 .5 .5 .2 .2\n', [2]), '')
        with self.assertRaises(ValueError): s.specialist_labels('6 .5 .5 .2 .2\n', [2])

    def test_training_subset_never_uses_heldouts_and_keeps_all_positives(self):
        rows = [dict(split='train', image_path=str(i), image_sha256=str(i), counts={'2': int(i < 2)})
                for i in range(5)]
        rows += [dict(split='val', image_path='val', counts={'2': 1}),
                 dict(split='test', image_path='test', counts={'2': 1})]
        subset = s.training_subset(rows, [2])
        self.assertEqual(len(subset), 4)
        self.assertTrue({'0', '1'} <= {r['image_path'] for r in subset})
        self.assertEqual({r['split'] for r in subset}, {'train'})
        self.assertEqual(subset, s.training_subset(rows, [2]))

    def test_hybrid_replaces_target_predictions_without_duplicates(self):
        base = {'a': [[.1,.1,.3,.3,.9,0], [.2,.2,.4,.4,.8,2]]}
        expert = {'a': [[.3,.3,.5,.5,.95,2]]}
        merged = s.merge_owned_predictions(base, [{'classes': ['utensil'], 'predictions': expert}])
        self.assertEqual(merged['a'], [base['a'][0], expert['a'][0]])
        self.assertEqual(len(base['a']), 2)
        for invalid in ([{'classes':['utensil'],'predictions':{'b':expert['a']}}],
                        [{'classes':['utensil'],'predictions':base}],
                        [{'classes':['bottle'],'predictions':expert}],
                        [{'classes':['utensil'],'predictions':expert}]*2):
            with self.assertRaises(ValueError): s.merge_owned_predictions(base, invalid)

    def test_specialist_acceptance_requires_target_gain_and_no_ap50_collapse(self):
        baseline = {'utensil': {'AP50': .5, 'AP50_95': .3}}
        good = {'utensil': {'AP50': .52, 'AP50_95': .33}}
        self.assertTrue(s.accept_specialist(baseline, good, ['utensil']))
        self.assertFalse(s.accept_specialist(baseline, baseline, ['utensil']))
        bad = {'utensil': {'AP50': .4, 'AP50_95': .4}}
        self.assertFalse(s.accept_specialist(baseline, bad, ['utensil']))
        bad['utensil']['AP50'] = float('nan')
        with self.assertRaises(ValueError): s.accept_specialist(baseline, bad, ['utensil'])

    def test_real_view_keeps_validation_images_and_rejects_label_tampering(self):
        import v5_dataset as v
        from test_v5_dataset import archive_fixture
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archive_fixture(root / 'source.zip')
            rows, contents, source = v.read_archive(root / 'source.zip', expected_images=3)
            for row in rows:
                row['label_text'] = ''.join(f'{i} .5 .5 .2 .2\n' for i in range(6))
                row['counts'] = {i: 1 for i in range(6)}
            dataset = root / 'base'
            v.materialize(rows, contents, source, ['train','val','test'], [0,1,2], dataset, [])
            view = root / 'utensil'
            receipt = s.build_view(dataset, view, [2])
            self.assertEqual(receipt['counts']['val_images'], 1)
            self.assertEqual(receipt['test_images_used'], 0)
            self.assertEqual((view / 'labels/train/00000.txt').read_text(), '0 .5 .5 .2 .2\n')
            self.assertEqual((view / 'images/val/00001.png').read_bytes(),
                             (dataset / 'images/val/00001.png').read_bytes())
            s.verify_view(view)
            (view / 'labels/train/00000.txt').write_text('')
            with self.assertRaisesRegex(ValueError, 'Changed'):
                s.verify_view(view)

    @unittest.skipUnless(importlib.util.find_spec('ultralytics'), 'Ultralytics unavailable')
    def test_native_specialist_validation_runs_and_maps_metric_class(self):
        import torch
        from ultralytics.nn.tasks import DetectionModel
        import v5_dataset as v
        from test_v5_dataset import archive_fixture
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archive_fixture(root / 'source.zip')
            rows, contents, source = v.read_archive(root / 'source.zip', expected_images=3)
            for row in rows:
                row['label_text'] = ''.join(f'{i} .5 .5 .2 .2\n' for i in range(6))
                row['counts'] = {i:1 for i in range(6)}
            v.materialize(rows, contents, source, ['train','val','test'], [0,1,2], root/'base', [])
            s.build_view(root/'base', root/'view', [2])
            model = DetectionModel('yolo11n.yaml', nc=1, verbose=False)
            model.names = {0:'utensil'}
            model.args = {'task':'detect', 'imgsz':800}
            torch.save({'model':model.half(), 'train_args':model.args}, root/'tiny.pt')
            metrics, predictions = s.evaluate_checkpoint(root/'tiny.pt', root/'view', root/'evaluation', device='cpu')
            self.assertEqual(set(metrics['per_class']), {'utensil'})
            self.assertEqual(set(predictions), {'00001'})
            self.assertTrue((root/'evaluation/report.json').is_file())
            loaded, _ = s.load_evaluation(root/'evaluation', root/'tiny.pt', root/'view')
            self.assertEqual(loaded['per_class'], metrics['per_class'])
            (root/'evaluation/predictions.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Changed'):
                s.load_evaluation(root/'evaluation', root/'tiny.pt', root/'view')


if __name__ == '__main__': unittest.main()
