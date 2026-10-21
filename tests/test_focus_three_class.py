import sys
import unittest
import json
import tempfile
import hashlib
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from focus_three_class import target_rows, repair_groups, audit, fingerprint, minimum_class_callback, profile_options
from focus_three_class import capture_group, apply_manual_reviews, validate_anchors, continuation_checkpoint, continuation_output
import yaml


class FocusTests(unittest.TestCase):
    def test_latest_resume_cannot_silently_rewind_completed_epochs(self):
        from focus_three_class import validate_resume_epoch
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            weights = run / 'train/weights'
            weights.mkdir(parents=True)
            (run / 'train/results.csv').write_text('epoch,time\n8,150\n9,312\n')
            with self.assertRaisesRegex(ValueError, 'behind|mismatch'):
                validate_resume_epoch(weights / 'last_resume.pt', dict(epoch=7))
            validate_resume_epoch(weights / 'last_resume.pt', dict(epoch=8))
            # A deliberately named earlier checkpoint remains available for controlled branches.
            validate_resume_epoch(weights / 'best_min_ap50.pt', dict(epoch=5))

    def test_continuation_can_replay_only_the_complete_audited_training_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            data = root / 'data'
            data.mkdir()
            train_a, train_b = data / 'images/train/a.jpg', data / 'images/train/b.jpg'
            validation = data / 'images/val/c.jpg'
            manifest = dict(records=[dict(name='a.jpg', split='train'), dict(name='b.jpg', split='train'),
                                     dict(name='c.jpg', split='val')])
            views = data / 'train-replay.txt'
            config = data / 'data-replay.yaml'
            config.write_text(yaml.safe_dump(dict(path=str(data), train=views.name, val='images/val',
                                                  names=['plate', 'wrapper', 'bottle'])))
            source = dict(epoch=6, optimizer={'state': {}}, train_args={'epochs': 80, 'data': 'original.yaml'})
            views.write_text('\n'.join(map(str, [train_a, train_b, train_b])))
            clone = continuation_checkpoint(source, root / 'output', data, 80,
                                            training_config=config, manifest=manifest)
            self.assertEqual(clone['train_args']['data'], str(config))
            self.assertEqual(source['train_args']['data'], 'original.yaml')
            for paths in ([train_a, validation], [train_a], [train_a, train_b, root / 'unknown.jpg']):
                views.write_text('\n'.join(map(str, paths)))
                with self.assertRaises(ValueError):
                    continuation_checkpoint(source, root / 'output', data, 80,
                                            training_config=config, manifest=manifest)

    def test_interrupted_resume_copy_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            last = root / 'last.pt'
            last.write_bytes(b'complete-new-checkpoint')
            history = root / 'results.csv'
            history.write_text('epoch\n8\n')
            latest = root / 'train/weights/last_resume.pt'
            latest.parent.mkdir(parents=True)
            latest.write_bytes(b'complete-previous-checkpoint')
            trainer = SimpleNamespace(epoch=7, csv=history, last=last)
            callback = minimum_class_callback(root, [1, 1, 1], .8, preserve_resume=True)

            def interrupted_copy(source, destination):
                Path(destination).write_bytes(b'truncated')
                raise OSError('simulated interrupted checkpoint copy')

            with patch('focus_three_class.shutil.copy2', side_effect=interrupted_copy):
                with self.assertRaisesRegex(OSError, 'interrupted'):
                    callback(trainer)
            self.assertEqual(latest.read_bytes(), b'complete-previous-checkpoint')
            self.assertEqual(last.read_bytes(), b'complete-new-checkpoint')
            self.assertEqual(sorted(p.name for p in latest.parent.iterdir()), ['last_resume.pt'])

    def test_reviewed_scene_members_cannot_cross_splits(self):
        rows = ['0 0.2 0.2 0.1 0.1', '1 0.5 0.5 0.1 0.1', '2 0.8 0.8 0.1 0.1']
        records = [dict(name=n, split=s, group=n, sha=n, pixels=n, rows=rows,
                        verified_absence=False, scene_group='same-table')
                   for n, s in [('a', 'train'), ('b', 'val')]]
        with self.assertRaisesRegex(ValueError, 'scene_group'):
            audit(records, 2)
        records[1]['frozen_val'] = True
        repair_groups(records)
        self.assertEqual([r['split'] for r in records], ['val', 'val'])

    def test_scene_reviews_require_exact_membership_and_image_hashes(self):
        from focus_three_class import scene_reviews
        records = [dict(name=n, sha=n) for n in ('a', 'b')]
        reviews = [dict(scene_id='table', reviewer='visual-review', reason='Same bottle and counter',
                        members=[dict(name=n, image_sha256=n) for n in ('a', 'b')])]
        scene_reviews(records, reviews, apply=True)
        self.assertEqual(records[0]['scene_group'], 'table')
        scene_reviews(records, reviews)
        records[0]['sha'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'hash'):
            scene_reviews(records, reviews)
        records[0]['sha'] = 'a'
        records[1].pop('scene_group')
        with self.assertRaisesRegex(ValueError, 'membership'):
            scene_reviews(records, reviews)

    def test_rebased_validation_rejects_legacy_checkpoint(self):
        from focus_three_class import validate_clean_source, PROJECT_ROOT
        manifest = dict(validation_protocol='new-captures')
        validate_clean_source(manifest, PROJECT_ROOT / 'model/weights/yolo11n.pt')
        with self.assertRaisesRegex(ValueError, 'fresh pretrained'):
            validate_clean_source(manifest, Path('old-best.pt'))
        with self.assertRaisesRegex(ValueError, 'protocol'):
            validate_clean_source(manifest, Path('old-best.pt'), dict(dataset_fingerprint='old'))
        validate_clean_source(manifest, Path('clean-last.pt'), dict(validation_protocol='new-captures'))

    def test_external_build_preserves_existing_images_labels_and_validation(self):
        from PIL import Image
        from focus_three_class import build_external, digest, pixels, audit, NAMES
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            base, output = root / 'baseline', root / 'supplement'
            records = []
            for split, color, rows in [('train', 'red', ['0 0.2 0.2 0.1 0.1', '1 0.5 0.5 0.1 0.1', '2 0.8 0.8 0.1 0.1']),
                                       ('train', 'green', []),
                                       ('val', 'blue', ['0 0.2 0.2 0.1 0.1', '1 0.5 0.5 0.1 0.1', '2 0.8 0.8 0.1 0.1'])]:
                name = f'base__{color}.jpg'
                image, label = base / 'images' / split / name, base / 'labels' / split / (Path(name).stem + '.txt')
                image.parent.mkdir(parents=True, exist_ok=True)
                label.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (20, 20), color).save(image)
                label.write_text('\n'.join(rows))
                records.append(dict(name=name, split=split, original_split=split, group='base__' + name.split('__')[1],
                                    sha=digest(image), pixels=pixels(image), rows=rows, verified_absence=not rows,
                                    review_status='signed_off', input_label_sha256=digest(label), override_applied=False))
            (base / 'data.yaml').write_text(yaml.safe_dump(dict(path=str(base), train='images/train', val='images/val', names=NAMES)))
            (base / 'manifest.json').write_text(json.dumps(dict(selected=3, records=records, source='fixture',
                                                               counts=audit(records, 3), fingerprint=fingerprint(records))))
            staging = root / 'data/focus-three-class-improvement/bottle-candidates'
            staging.mkdir(parents=True)
            extra = staging / 'extra.jpg'
            Image.new('RGB', (20, 20), 'yellow').save(extra)
            (staging / 'candidates.json').write_text(json.dumps([dict(id='extra', path=str(extra), sha256=digest(extra),
                status='pending_full_three_class_review', url='https://example.org/extra.jpg')]))
            review = staging / 'reviews.json'
            review.write_text(json.dumps([dict(candidate_id='extra', session='newscene', image_sha256=digest(extra),
                full_three_class_review=True, duplicate_reviewed=True, reviewer='reviewer', reason='Full review',
                rows=['2 0.5 0.5 0.4 0.4'])]))
            with patch('focus_three_class.PROJECT_ROOT', root):
                build_external(SimpleNamespace(frozen_data=base, data=output, review_files=[review]))
            for r in records:
                for kind, suffix in [('images', '.jpg'), ('labels', '.txt')]:
                    relative = Path(kind) / r['split'] / Path(r['name']).with_suffix(suffix)
                    self.assertEqual((base / relative).read_bytes(), (output / relative).read_bytes())
            self.assertEqual(len(list((output / 'images/val').glob('*.jpg'))), 1)

    def test_external_review_requires_complete_hash_bound_nonduplicate_evidence(self):
        from PIL import Image
        from focus_three_class import external_record, digest, pixels
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / 'photo.jpg'
            Image.new('RGB', (20, 10), 'green').save(image)
            candidate = dict(id='sample', path=str(image), sha256=digest(image),
                             status='pending_full_three_class_review', url='https://example.org/photo.jpg')
            review = dict(candidate_id='sample', session='scene1', image_sha256=digest(image),
                          full_three_class_review=True, duplicate_reviewed=True,
                          reviewer='reviewer', reason='Full scene reviewed',
                          rows=['2 0.5 0.5 0.4 0.4'])
            record = external_record(review, candidate, set(), set())
            self.assertEqual(record['split'], 'train')
            self.assertEqual(record['capture'], 'external_bottle__scene1')
            self.assertEqual(record['rows'], review['rows'])
            for change in ({'full_three_class_review': False}, {'duplicate_reviewed': False},
                           {'image_sha256': 'stale'}, {'rows': []}, {'session': '../escape'},
                           {'rows': ['2 0.5 0.5 2 2']}):
                with self.assertRaises(ValueError):
                    external_record({**review, **change}, candidate, set(), set())
            for change in ({'status': 'possible_existing_duplicate_hold'}, {'sha256': 'stale'}):
                with self.assertRaises(ValueError):
                    external_record(review, {**candidate, **change}, set(), set())
            with self.assertRaises(ValueError):
                external_record(review, candidate, {digest(image)}, set())
            with self.assertRaises(ValueError):
                external_record(review, candidate, set(), {pixels(image)})
            image.write_bytes(image.read_bytes() + b'SEFT metadata trailer')
            candidate['sha256'] = digest(image)
            review['image_sha256'] = digest(image)
            with self.assertRaisesRegex(ValueError, 'JPEG'):
                external_record(review, candidate, set(), set())

    def test_existing_training_capture_cannot_move_into_expanded_validation(self):
        records = [dict(name='a', split='train', frozen_train=True, group='a', sha='a', pixels='a', capture='meal'),
                   dict(name='b', split='val', group='b', sha='b', pixels='b', capture='meal'),
                   dict(name='c', split='val', group='c', sha='c', pixels='c', capture='meal')]
        repair_groups(records)
        self.assertEqual([r['split'] for r in records], ['train'] * 3)
        records[1]['frozen_val'] = True
        with self.assertRaisesRegex(ValueError, 'Conflicting frozen'):
            repair_groups(records)

    def test_continuation_output_is_new_direct_child_of_improvement_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            improvement = root / 'model/focus-three-class-improvement'
            improvement.mkdir(parents=True)
            source = improvement / 'source'
            source.mkdir()
            with patch('focus_three_class.PROJECT_ROOT', root):
                continuation_output(improvement / 'new-run', source)
                for output in (source, root / 'model/weights', source / 'nested',
                               improvement / 'source/../source'):
                    with self.assertRaises(ValueError):
                        continuation_output(output, source)
                linked = improvement / 'linked'
                linked.symlink_to(improvement / 'missing')
                with self.assertRaises(ValueError):
                    continuation_output(linked, source)

    def test_continuation_clone_preserves_source_and_rejects_stripped(self):
        original = dict(epoch=6, optimizer={'state': {}}, train_args={'epochs': 80, 'time': None, 'data': '/audited/data-replay.yaml'})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clone = continuation_checkpoint(original, root, root / 'data', 80)
            self.assertEqual(original['train_args'], {'epochs': 80, 'time': None, 'data': '/audited/data-replay.yaml'})
            self.assertEqual(clone['train_args']['data'], '/audited/data-replay.yaml')
            self.assertEqual(clone['train_args']['epochs'], 80)
            self.assertIsNone(clone['train_args']['time'])
            self.assertEqual(clone['train_args']['save_dir'], str(root / 'train'))
            for change in ({'optimizer': None}, {'epoch': -1}, {'epoch': 79},
                           {'train_args': {'epochs': 7, 'time': .15}}):
                with self.assertRaises(ValueError):
                    continuation_checkpoint({**original, **change}, root, root / 'data', 80)

    def test_continuation_wall_limit_latest_guard_and_inherited_best(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history, last = root / 'results.csv', root / 'last.pt'
            history.write_text('epoch\n8\n')
            last.write_bytes(b'epoch-eight')
            trainer = SimpleNamespace(epoch=7, csv=history, last=last, stop=False,
                                      epochs=80, args=SimpleNamespace(time=None),
                                      validator=SimpleNamespace(metrics=SimpleNamespace(
                                          nt_per_class=[1, 1, 1], box=SimpleNamespace(
                                              ap_class_index=[0, 1, 2], ap50=[.8, .8, .8]))))
            inherited_best = root / 'train/weights/best_min_ap50.pt'
            inherited_best.parent.mkdir(parents=True)
            inherited_best.write_bytes(b'better-parent')
            selection = root / 'minimum_class_selection.json'
            selection.write_text('{"minimum_AP50": 0.85}')
            callback = minimum_class_callback(root, [1, 1, 1], .9, initial_score=.85,
                                               minutes=1, preserve_resume=True)
            with patch('focus_three_class.time.monotonic', return_value=0):
                callback.start(trainer)
            with patch('focus_three_class.time.monotonic', return_value=61):
                callback(trainer)
            self.assertTrue(trainer.stop)
            self.assertEqual(trainer.epochs, 80)
            self.assertIsNone(trainer.args.time)
            latest = root / 'train/weights/last_resume.pt'
            self.assertEqual(latest.read_bytes(), b'epoch-eight')
            self.assertEqual(inherited_best.read_bytes(), b'better-parent')
            self.assertEqual(json.loads(selection.read_text())['minimum_AP50'], .85)
            trainer.epoch = 8
            last.write_bytes(b'final-stripped')
            callback(trainer)
            self.assertEqual(latest.read_bytes(), b'epoch-eight')

    def test_frozen_anchor_rejects_changed_image_with_same_name_and_rows(self):
        anchor = dict(name='same.jpg', split='val', sha='original', pixels='original-pixels',
                      rows=['0 0.5 0.5 0.2 0.2'])
        baseline = dict(records=[anchor])
        validate_anchors([anchor.copy()], baseline)
        for field in ('sha', 'pixels', 'rows'):
            changed = {**anchor, field: [] if field == 'rows' else 'changed'}
            with self.assertRaisesRegex(ValueError, 'Frozen validation evidence changed'):
                validate_anchors([changed], baseline)

    def test_stack_policy_allows_only_plate_relabel_on_same_frozen_image(self):
        plate = '0 0.5 0.5 0.2 0.2'
        wrapper = '1 0.4 0.4 0.1 0.1'
        bottle = '2 0.3 0.3 0.1 0.1'
        anchor = dict(name='anchor.jpg', split='val', sha='same', pixels='same-pixels',
                      rows=[plate, wrapper, bottle])
        changed = {**anchor, 'rows': ['0 0.5 0.5 0.4 0.4', wrapper, bottle]}
        baseline = dict(records=[anchor])
        with self.assertRaisesRegex(ValueError, 'Frozen validation evidence changed'):
            validate_anchors([changed], baseline)
        self.assertEqual(validate_anchors([changed], baseline, allow_plate_relabel=True),
                         [dict(name='anchor.jpg', original_rows=anchor['rows'], new_rows=changed['rows'])])
        for rows in ([plate, bottle], [plate, wrapper],
                     [plate, wrapper, '2 0.6 0.6 0.1 0.1']):
            with self.assertRaisesRegex(ValueError, 'Frozen validation evidence changed'):
                validate_anchors([{**anchor, 'rows': rows}], baseline, allow_plate_relabel=True)
        for key in ('sha', 'pixels'):
            with self.assertRaisesRegex(ValueError, 'Frozen validation evidence changed'):
                validate_anchors([{**changed, key: 'different'}], baseline, allow_plate_relabel=True)

    def test_manual_review_hashes_duplicates_and_rows(self):
        rows = ['0 0.5 0.5 0.2 0.2']
        record = dict(name='a.jpg', sha='image', rows=rows)
        review = dict(name='a.jpg', status='verified_unchanged', rows=rows,
                      image_sha256='image', input_rows_sha256=hashlib.sha256(
                          json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                      reviewer='native', reason='inspected')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'review.json'
            for changes, error in [({'image_sha256': 'stale'}, 'hash'),
                                   ({'input_rows_sha256': 'stale'}, 'hash'),
                                   ({'rows': ['3 0.5 0.5 0.2 0.2']}, 'class')]:
                path.write_text(json.dumps([{**review, **changes}]))
                with self.assertRaisesRegex(ValueError, error):
                    apply_manual_reviews([record.copy()], [path])
            path.write_text(json.dumps([review, review]))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                apply_manual_reviews([record.copy()], [path])

    def test_capture_groups_transitive_and_frozen_validation(self):
        first = capture_group('school_lunch__train__010136000.jpg')
        self.assertEqual(first, capture_group('school_lunch__val__010136001.jpg'))
        self.assertNotEqual(first, capture_group('school_lunch__train__010137000.jpg'))
        self.assertEqual(capture_group('waste_detection_v3__train__video-one-0001_jpg.rf.abc.jpg'),
                         capture_group('waste_detection_v3__val__video-one-0009_jpg.rf.def.jpg'))
        records = [dict(name=str(i), split='train', group=str(i), sha=str(i),
                        pixels=str(i), capture=first) for i in range(3)]
        records[2]['frozen_val'] = True
        repair_groups(records)
        self.assertEqual([r['split'] for r in records], ['val'] * 3)

    def test_capture_leakage_is_rejected(self):
        rows = [f'{i} 0.5 0.5 0.2 0.2' for i in range(3)]
        records = [dict(name=str(i), split=split, group=str(i), sha=str(i),
                        pixels=str(i), capture='meal', rows=rows,
                        verified_absence=False) for i, split in enumerate(('train', 'val'))]
        with self.assertRaisesRegex(ValueError, 'capture'):
            audit(records, expected_count=2)
        records[0]['frozen_val'] = True
        records[1]['capture'] = 'other'
        records.append(dict(records[1], name='third', group='third', sha='third',
                            pixels=records[0]['pixels'], capture='other'))
        repair_groups(records)
        self.assertEqual([r['split'] for r in records], ['val'] * 3)

    def test_optional_capture_fingerprint_keeps_original_encoding(self):
        record = dict(name='a', split='train', sha='s', rows=[], group='g', pixels='p',
                      verified_absence=True, review_status='ok', original_split='train',
                      input_label_sha256='l', override_applied=False)
        expected = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
        self.assertEqual(fingerprint([record]), expected)
        self.assertNotEqual(fingerprint([{**record, 'capture': 'meal'}]), expected)
        self.assertNotEqual(fingerprint([{**record, 'scene_group': 'reviewed-table'}]), expected)

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
