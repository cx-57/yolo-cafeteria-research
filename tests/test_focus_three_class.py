import sys
import unittest
import json
import tempfile
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from focus_three_class import target_rows, repair_groups, audit, fingerprint, minimum_class_callback, profile_options
import yaml


class FocusTests(unittest.TestCase):
    def test_remap_preserves_coordinates_and_all_boxes(self):
        rows = ['1 0.5 0.5 0.2 0.2', '8 0.4 0.4 0.1 0.1',
                '9 0.3 0.3 0.2 0.2', '0 0.5 0.5 0.1 0.1']
        self.assertEqual(target_rows(rows), [str(i) + row[1:] for i, row in enumerate(rows[:3])])

    def test_bad_box_stops(self):
        for row in ['1 nan 0.5 0.2 0.2', '8 0.95 0.5 0.2 0.2', '9 0.5 0.5 0 0.2']:
            with self.assertRaises(ValueError):
                target_rows([row])

    def test_transitive_duplicate_groups_choose_fewest_moves(self):
        records = [dict(name='a', split='val', group='g1', pixels='p1', sha='s1'),
                   dict(name='b', split='train', group='g1', pixels='p2', sha='s2'),
                   dict(name='c', split='val', group='g2', pixels='p2', sha='s3')]
        moves = repair_groups(records)
        self.assertEqual([r['split'] for r in records], ['val'] * 3)
        self.assertEqual(len(moves), 1)

    def test_empty_label_requires_explicit_absence_verification(self):
        record = dict(name='a', split='train', group='g', sha='s', pixels='p',
                      rows=[], verified_absence=False)
        with self.assertRaisesRegex(ValueError, 'Unverified negative'):
            audit([record])

    def test_fingerprint_binds_leakage_and_review_evidence(self):
        record = dict(name='a', split='train', sha='s', rows=[], group='g', pixels='p',
                      verified_absence=True, review_status='corrected_and_signed_off',
                      original_split='train', input_label_sha256='l', override_applied=True)
        original = fingerprint([record])
        for field, replacement in [('group', 'other'), ('pixels', 'other'),
                                   ('verified_absence', False)]:
            self.assertNotEqual(fingerprint([{**record, field: replacement}]), original)

    def test_expansion_requires_explicit_audited_count(self):
        rows = [f'{i} 0.5 0.5 0.2 0.2' for i in range(3)]
        records = [dict(name=str(i), split='train' if i < 600 else 'val',
                        group=str(i), sha=str(i), pixels=str(i), rows=rows,
                        verified_absence=False) for i in range(619)]
        with self.assertRaisesRegex(ValueError, '613'):
            audit(records)
        self.assertEqual(audit(records, expected_count=619)['train']['images'], 600)

    def test_checkpoint_selection_protects_weak_class_and_skips_final_revalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            last = root / 'last.pt'
            history = root / 'results.csv'
            callback = minimum_class_callback(root, [89, 8, 36], .8)
            trainer = SimpleNamespace(epoch=0, csv=history, last=last, stop=False)
            for epoch, scores in enumerate(([.99, .99, .61], [.81, .82, .83])):
                last.write_bytes(f'epoch-{epoch}'.encode())
                history.write_text(f'epoch\n{epoch+1}\n')
                trainer.epoch = epoch
                trainer.validator = SimpleNamespace(metrics=SimpleNamespace(
                    nt_per_class=[89, 8, 36], box=SimpleNamespace(
                        ap_class_index=[0, 1, 2], ap50=scores)))
                callback(trainer)
            selected = root / 'train/weights/best_min_ap50.pt'
            self.assertEqual(selected.read_bytes(), b'epoch-1')
            self.assertTrue(trainer.stop)
            self.assertEqual(json.loads((root / 'minimum_class_selection.json').read_text())['epoch'], 2)
            trainer.epoch = 2  # Native final evaluation reports another checkpoint, not last.pt.
            last.write_bytes(b'wrong-final-weights')
            trainer.validator.metrics.box.ap50 = [.99, .99, .99]
            callback(trainer)
            self.assertEqual(selected.read_bytes(), b'epoch-1')

    def test_recipe_cannot_train_on_validation_or_resume_into_original_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            listing = root / 'train.txt'
            config = root / 'data.yaml'
            config.write_text(yaml.safe_dump(dict(path=str(root), train=str(listing),
                                                 val='images/val', names=['plate', 'wrapper', 'bottle'])))
            recipe = root / 'recipe.json'
            options = dict(model='source.pt', data=str(config))
            manifest = dict(records=[dict(name='training.jpg', split='train')])
            listing.write_text(str(root / 'images/val/leaked.jpg') + '\n')
            recipe.write_text(json.dumps(dict(training_args=options)))
            with self.assertRaisesRegex(ValueError, 'validation training images'):
                profile_options(recipe, root, manifest)
            listing.write_text(str(root / 'images/train/training.jpg') + '\n')
            recipe.write_text(json.dumps(dict(training_args={**options, 'resume': True})))
            with self.assertRaisesRegex(ValueError, 'resume'):
                profile_options(recipe, root, manifest)
            hidden = root / 'hidden.yaml'
            hidden.write_text('resume: /original/run/last.pt\nsingle_cls: true\n')
            recipe.write_text(json.dumps(dict(training_args={**options, 'cfg': str(hidden)})))
            with self.assertRaisesRegex(ValueError, 'cfg'):
                profile_options(recipe, root, manifest)


if __name__ == '__main__':
    unittest.main()
