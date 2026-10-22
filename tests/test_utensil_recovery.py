"""Safety and provenance checks for the separate recovery benchmark."""
import importlib.util
import hashlib
import copy
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
import utensil_recovery as recovery


ML_AVAILABLE = bool(importlib.util.find_spec('torch') and importlib.util.find_spec('ultralytics'))


class RecoveryTests(unittest.TestCase):
    def test_removing_last_target_requires_explicit_verified_absence(self):
        image = self.source / 'images/train/utensil__stock.jpg'
        label = recovery.label_path(image)
        review = dict(name=image.name, image_sha256=recovery.digest(image),
            input_label_sha256=recovery.digest(label), reviewer='test', reason='Printed artwork',
            remove_rows=label.read_text().splitlines(), additions=[])
        self.reviews.write_text(json.dumps([review]))
        with self.assertRaisesRegex(ValueError, 'absence|negative'):
            recovery.project(self.source/'data.yaml',self.reviews)
        review['verified_absent']=[3]
        self.reviews.write_text(json.dumps([review]))
        records,_,_=recovery.project(self.source/'data.yaml',self.reviews)
        self.assertEqual(next(r['rows'] for r in records if r['name']==image.name),[])

    def test_full_schedule_does_not_stop_on_a_provisional_target_score(self):
        output = self.root / 'full-run'; output.mkdir()
        csv = self.root / 'full-results.csv'; csv.write_text('epoch\n1\n')
        last = self.root / 'last.pt'; last.write_bytes(b'checkpoint')
        counts = dict.fromkeys(self.names, 0); counts.update(utensil=30, condiment_packet=30)
        self.assertIn('stop_at_target', __import__('inspect').signature(recovery.checkpoint_callback).parameters)
        callback = recovery.checkpoint_callback(output, counts,
            targets=('utensil', 'condiment_packet'), stop_at_target=False)
        trainer = SimpleNamespace(epoch=0, csv=csv, last=last, stop=False,
            validator=SimpleNamespace(metrics=SimpleNamespace(nt_per_class=[counts[n] for n in self.names],
                box=SimpleNamespace(ap_class_index=[3,7], ap50=[.9,.9]))))
        callback.start(trainer); callback(trainer)
        self.assertFalse(trainer.stop)
        self.assertEqual(json.loads((output/'utensil_selection.json').read_text())['epoch'],1)

    @unittest.skipUnless(ML_AVAILABLE, 'requires ML dependencies; covered by ml-integration')
    def test_full_schedule_disables_native_early_stopping_without_changing_loader(self):
        from ultralytics.utils.torch_utils import EarlyStopping
        class Base:
            def _setup_train(self):
                self.args = SimpleNamespace(patience=15)
                self.stopper = EarlyStopping(patience=15)
        self.assertIn('full_schedule', __import__('inspect').signature(recovery.make_recovery_trainer).parameters)
        with patch('partial_labels.make_trainer', return_value=Base):
            trainer = recovery.make_recovery_trainer(self.root/'policy', full_schedule=True)()
        trainer._setup_train()
        self.assertEqual(trainer.args.patience,0)
        self.assertFalse(trainer.stopper(100, .0))

    def test_external_training_packet_can_preserve_reviewed_eating_cutlery(self):
        image = self.root / 'packet-meal.jpg'
        Image.new('RGB', (20, 15), (91, 82, 73)).save(image)
        external = self.root / 'packet-review.json'
        review = dict(name='packet_extra__meal.jpg', image=str(image),
            image_sha256=recovery.digest(image), split='train', capture='photographer_new',
            original_group='new_meal', rows=['7 .2 .2 .2 .2'], reviewer='test',
            reason='Sauce packet and eating fork reviewed', url='https://example.org/photo',
            license='CC BY', artist='One', exhaustive_packet_review=True,
            exhaustive_utensil_review=True, utensil_additions=['3 .7 .7 .2 .2'])
        external.write_text(json.dumps([review]))
        records, _, _ = recovery.project(self.source / 'data.yaml', self.reviews, external)
        self.assertEqual(next(r['rows'] for r in records if r['name'] == review['name']),
                         ['7 .2 .2 .2 .2', '3 .7 .7 .2 .2'])
        for change in [dict(exhaustive_utensil_review=False),
                       dict(utensil_additions=['1 .7 .7 .2 .2']),
                       dict(split='val', verified_utensil_absence=True)]:
            external.write_text(json.dumps([{**review, **change}]))
            with self.subTest(change=change), self.assertRaises(ValueError):
                recovery.project(self.source / 'data.yaml', self.reviews, external)

    def test_training_correction_can_replace_a_wrong_target_without_changing_source(self):
        image = self.source / 'images/train/utensil__stock.jpg'
        label = recovery.label_path(image)
        original = label.read_text()
        review = dict(name=image.name, image_sha256=recovery.digest(image),
            input_label_sha256=recovery.digest(label), reviewer='visual review',
            reason='Replace a handle-only box with the complete eating spoon',
            remove_rows=original.splitlines(), additions=['3 .6 .5 .5 .4'])
        self.reviews.write_text(json.dumps([review]))
        result = self.build()
        record = next(r for r in result['records'] if r['name'] == image.name)
        self.assertEqual(record['rows'], ['3 .6 .5 .5 .4'])
        self.assertEqual(label.read_text(), original)

    def test_training_corrections_reject_non_targets_unknown_rows_and_heldout_images(self):
        for name, removed in [('utensil__stock.jpg', ['3 .1 .1 .1 .1']),
                              ('utensil__stock.jpg', ['1 .5 .5 .4 .4']),
                              ('utensil__WhatsApp-Image-2022-01-27-at-a.jpg', ['3 .5 .5 .4 .4'])]:
            image = self.source / 'images/train' / name
            self.reviews.write_text(json.dumps([dict(name=name, image_sha256=recovery.digest(image),
                input_label_sha256=recovery.digest(recovery.label_path(image)), reviewer='test',
                reason='test', remove_rows=removed, additions=[])]))
            with self.subTest(name=name, removed=removed), self.assertRaises(ValueError):
                recovery.project(self.source / 'data.yaml', self.reviews)

    @unittest.skipUnless(ML_AVAILABLE, 'requires ML dependencies; covered by ml-integration')
    def test_reviewed_packet_absence_adds_negative_gradient_only_for_that_photo(self):
        import torch
        from partial_labels import CoveragePolicy, CoverageMaskedBCE
        self.assertTrue(hasattr(recovery, 'ReviewedCoverage'), 'Need verified photo-level coverage')
        base = CoveragePolicy(tuple(self.names), {'plate': (1,)})
        policy = recovery.ReviewedCoverage(base, {'plate__reviewed.jpg': (7,)})
        mask = policy.known_mask(['plate__reviewed.jpg', 'plate__unknown.jpg'], 2, 'cpu')
        bce = CoverageMaskedBCE(); bce.set_known_mask(mask)
        logits = torch.zeros(2, 1, 11, requires_grad=True)
        bce(logits, torch.zeros_like(logits)).sum().backward()
        self.assertGreater(logits.grad[0, 0, 7].item(), 0)
        self.assertEqual(logits.grad[1, 0, 7].item(), 0)
        self.assertEqual(logits.grad[0, 0, 3].item(), 0)

    def test_photo_coverage_requires_train_membership_hashes_and_verified_absence(self):
        manifest = self.build()
        record = next(r for r in manifest['records'] if r['name'] == 'utensil__stock.jpg')
        evidence = self.root / 'coverage-reviews.json'
        policy = self.root / 'coverage.json'
        review = dict(name=record['name'], image_sha256=record['sha'], rows=record['rows'],
            known_classes=[3, 7], verified_absent=[7], reviewer='test', reason='Both classes inspected')
        def write_review(value):
            evidence.write_text(json.dumps([value]))
            policy.write_text(json.dumps(dict(image_reviews=dict(path=str(evidence),
                sha256=recovery.digest(evidence), dataset_fingerprint=manifest['fingerprint']))))
        self.assertTrue(hasattr(recovery, 'reviewed_coverage'))
        write_review(review)
        self.assertEqual(recovery.reviewed_coverage(policy, manifest), {record['name']: (3, 7)})
        for change in [dict(verified_absent=[]), dict(verified_absent=[3, 7]),
                       dict(image_sha256='wrong'), dict(rows=[]), dict(known_classes=[1]),
                       dict(name=next(r['name'] for r in manifest['records'] if r['split']=='val'))]:
            write_review({**review, **change})
            with self.subTest(change=change), self.assertRaises(ValueError):
                recovery.reviewed_coverage(policy, manifest)

    def test_nominal_batch_ablation_preserves_learning_state_and_parent(self):
        checkpoint = dict(epoch=14, optimizer={'state': {'step': 178}}, updates=178,
            ema={'learned': True}, train_args={'epochs': 80, 'data': 'old.yaml', 'nbs': 64, 'batch': 4})
        original = copy.deepcopy(checkpoint)
        try:
            clone = recovery.revision_checkpoint(checkpoint, self.root / 'ablation',
                self.root / 'new.yaml', nominal_batch=4)
        except TypeError as error:
            self.fail('Need an explicit optimizer-step-frequency experiment: ' + str(error))
        self.assertEqual(checkpoint, original)
        self.assertEqual(clone['train_args']['nbs'], 4)
        self.assertEqual(clone['train_args']['batch'], 4)
        for key in ('epoch', 'updates', 'optimizer', 'ema'):
            self.assertEqual(clone[key], checkpoint[key])

    @unittest.skipUnless(ML_AVAILABLE, 'requires ML dependencies; covered by ml-integration')
    def test_validation_batch_override_preserves_training_loader_and_coverage_factory(self):
        self.assertTrue(hasattr(recovery, 'make_recovery_trainer'))
        class Base:
            def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
                return (dataset_path, batch_size, rank, mode)
        with patch('partial_labels.make_trainer', return_value=Base) as factory:
            trainer = recovery.make_recovery_trainer(self.root / 'policy.json', validation_batch=4)()
        factory.assert_called_once_with(self.root / 'policy.json')
        self.assertIsInstance(trainer, Base)
        self.assertEqual(trainer.get_dataloader('val', 8, -1, 'val'), ('val', 4, -1, 'val'))
        self.assertEqual(trainer.get_dataloader('train', 4, -1, 'train'), ('train', 4, -1, 'train'))

    def test_multi_target_selection_uses_the_weaker_class(self):
        output = self.root / 'joint-model'
        output.mkdir()
        csv = self.root / 'joint-results.csv'
        last = self.root / 'joint-last.pt'
        counts = dict.fromkeys(self.names, 0)
        counts.update(utensil=30, condiment_packet=30)
        try:
            callback = recovery.checkpoint_callback(output, counts, targets=('utensil', 'condiment_packet'))
        except TypeError as error:
            self.fail('Joint goal must select for both classes: ' + str(error))
        trainer = SimpleNamespace(epoch=0, csv=csv, last=last, stop=False)
        callback.start(trainer)
        for epoch, scores in enumerate(([.8, .1], [.7, .6], [.9, .2])):
            trainer.epoch = epoch
            csv.write_text(f'epoch\n{epoch+1}\n')
            last.write_bytes(f'epoch{epoch+1}'.encode())
            trainer.validator = SimpleNamespace(metrics=SimpleNamespace(
                nt_per_class=[counts[n] for n in self.names],
                box=SimpleNamespace(ap_class_index=[3, 7], ap50=scores)))
            callback(trainer)
        self.assertEqual((output / 'train/weights/best_utensil.pt').read_bytes(), b'epoch2')
        self.assertFalse(trainer.stop)
        self.assertEqual(json.loads((output / 'utensil_selection.json').read_text())['selection_score'], .6)

    def test_offloaded_source_files_fail_before_a_blocking_read(self):
        self.assertTrue(hasattr(recovery, 'require_local'))
        with patch.object(Path, 'stat', return_value=SimpleNamespace(st_flags=0x40000000)):
            with self.assertRaisesRegex(ValueError, 'iCloud|offloaded'):
                recovery.require_local([self.root / 'placeholder.txt'])

    def test_external_packets_bind_pixels_review_and_capture_split(self):
        image = self.root / 'new-packet.jpg'
        Image.new('RGB', (20, 15), (91, 82, 73)).save(image)
        review = dict(name='packet_extra__commons_123.jpg', image=str(image),
            image_sha256=recovery.digest(image), split='val', capture='commons_author_one',
            original_group='commons_123', rows=['7 .5 .5 .4 .4'], reviewer='test',
            reason='All visible sauce sachets reviewed', url='https://example.org/photo',
            license='CC BY 4.0', artist='One', exhaustive_packet_review=True,
            verified_utensil_absence=True)
        extra = self.root / 'external.json'
        extra.write_text(json.dumps([review]))
        try:
            manifest = recovery.build(self.source / 'data.yaml', self.output, self.reviews,
                                      external=extra)
        except TypeError as error:
            self.fail('Need a hash-bound import for genuinely new packet contexts: ' + str(error))
        recovery.verify(self.output)
        self.assertEqual(manifest['counts']['val']['condiment_packet'], 1)
        self.assertEqual(recovery.digest(self.output / 'images/val' / review['name']),
                         review['image_sha256'])
        review['verified_utensil_absence'] = False
        extra.write_text(json.dumps([review]))
        with self.assertRaisesRegex(ValueError, 'utensil|absence'):
            recovery.project(self.source / 'data.yaml', self.reviews, extra)
        review['image_sha256'] = 'wrong'
        extra.write_text(json.dumps([review]))
        with self.assertRaises(ValueError):
            recovery.verify(self.output)

    def test_renamed_existing_photos_cannot_inflate_expanded_support(self):
        old = self.build()
        for split in ('train', 'val'):
            new = copy.deepcopy(old)
            original = next(r for r in old['records'] if r['split'] == split)
            duplicate = copy.deepcopy(original)
            duplicate.update(name='packet_extra__renamed.jpg', rows=['7 .5 .5 .4 .4'])
            new['records'].append(duplicate)
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, 'duplicate|new|overlap'):
                recovery.validate_expansion(old, new)

    def test_external_import_rejects_copy_of_old_capture_before_repair(self):
        image = self.source / 'images/train/utensil__stock.jpg'
        extra = self.root / 'renamed.json'
        extra.write_text(json.dumps([dict(name='packet_extra__renamed.jpg', image=str(image),
            image_sha256=recovery.digest(image), split='val', capture='pretend_new',
            original_group='pretend_original', rows=['7 .5 .5 .4 .4'], reviewer='test',
            reason='test', url='https://example.org/photo', license='CC0', artist='One',
            exhaustive_packet_review=True, verified_utensil_absence=True)]))
        with self.assertRaisesRegex(ValueError, 'duplicate|overlap|new'):
            recovery.build(self.source / 'data.yaml', self.output, self.reviews, external=extra)
        self.assertFalse(self.output.exists())

    def test_external_capture_cannot_move_an_old_training_photo_to_validation(self):
        self.add('train', 'utensil__another.jpg', '3 .5 .5 .4 .4', 8)
        name = 'utensil__IMG_20231109_123456.jpg'
        self.add('train', name, '3 .5 .5 .4 .4', 9)
        image = self.root / 'new.jpg'
        Image.new('RGB', (20, 15), (91, 82, 73)).save(image)
        extra = self.root / 'capture-overlap.json'
        extra.write_text(json.dumps([dict(name='packet_extra__new.jpg', image=str(image),
            image_sha256=recovery.digest(image), split='val', capture=recovery.capture(name),
            original_group='new_photo', rows=['7 .5 .5 .4 .4'], reviewer='test',
            reason='test', url='https://example.org/photo', license='CC0', artist='One',
            exhaustive_packet_review=True, verified_utensil_absence=True)]))
        with self.assertRaisesRegex(ValueError, 'capture|overlap'):
            recovery.build(self.source / 'data.yaml', self.output, self.reviews, external=extra)
        self.assertFalse(self.output.exists())

    def test_expansion_keeps_old_holdout_and_rejects_policy_drift(self):
        self.assertTrue(hasattr(recovery, 'validate_expansion'))
        old = self.build()
        new = copy.deepcopy(old)
        extra = copy.deepcopy(old['records'][0])
        extra.update(name='packet_extra__new.jpg', split='val', original_split='val',
            sha='new-sha', pixels='new-pixels', group='new-original', capture='new-capture',
            frozen_val=True, rows=['7 .5 .5 .2 .2'])
        new['records'].append(extra)
        self.assertEqual(recovery.validate_expansion(old, new)['new_images'], 1)
        next(r for r in new['records'] if r['split'] == 'val')['rows'].append('3 .8 .8 .1 .1')
        with self.assertRaises(ValueError):
            recovery.validate_expansion(old, new)
        self.assertTrue(hasattr(recovery, 'validate_policy_extension'))
        previous = {'schema_version': 1, 'names': self.names,
                    'source_known_classes': {'plate': [1]}}
        revised = copy.deepcopy(previous)
        revised['source_known_classes']['packet_extra'] = [7]
        recovery.validate_policy_extension(previous, revised)
        revised['source_known_classes']['plate'] = [1, 3]
        with self.assertRaises(ValueError):
            recovery.validate_policy_extension(previous, revised)

    @unittest.skipUnless(ML_AVAILABLE, 'requires ML dependencies; covered by ml-integration')
    def test_rebenchmark_selection_uses_fresh_score_instead_of_inherited_ap(self):
        import torch
        previous = self.build()
        model_root = self.root / 'models'
        parent, child = model_root / 'parent', model_root / 'rebench'
        weights = parent / 'train/weights'
        weights.mkdir(parents=True)
        listing = self.root / 'old-train.txt'
        listing.write_text('\n'.join(str(self.output / 'images/train' / r['name'])
            for r in previous['records'] if r['split'] == 'train') + '\n')
        config = self.root / 'old.yaml'
        config.write_text(yaml.safe_dump(dict(path=str(self.output), train=str(listing),
            val=str(self.output / 'images/val'), names=self.names)))
        binding = dict(dataset_fingerprint=previous['fingerprint'], training_yaml=str(config),
            training_yaml_sha256=recovery.digest(config), training_list_sha256=recovery.digest(listing),
            policy_sha256='policy', pretrained_sha256='pretrained', training_holds=None, epochs=80)
        (parent / 'experiment.json').write_text(json.dumps(dict(binding=binding)))
        checkpoint = dict(epoch=0, optimizer={'state': {}}, ema=None, updates=1,
            best_fitness=.99, stopper={'best_fitness': .99, 'best_epoch': 0},
            train_args={'epochs': 80, 'data': str(config)})
        source = weights / 'last_resume.pt'
        torch.save(checkpoint, source)
        (parent / 'train/results.csv').write_text('epoch\n1\n')
        selected_path = weights / 'best_utensil.pt'
        selected_path.write_bytes(source.read_bytes())
        selected = dict(epoch=1, utensil_AP50=.99, per_class_AP50={'utensil': .99},
            weights=str(selected_path), weights_sha256=recovery.digest(selected_path))
        (parent / 'utensil_selection.json').write_text(json.dumps(selected))
        revised = copy.deepcopy(previous)
        next(r for r in revised['records'] if r['split'] == 'val')['rows'].append('3 .8 .8 .1 .1')
        revised['fingerprint'] = 'revised-labels'
        new_binding = {**binding, 'dataset_fingerprint': 'revised-labels'}
        fresh = dict(per_class={'utensil': {'ap50': .31}, 'paper_boat': {'ap50': None}})
        source_hash = recovery.digest(source)
        # GPU evaluation is the only mocked boundary; all fork/provenance/file logic runs.
        with patch.object(recovery, 'OUTPUT', parent), patch('train.evaluate', return_value=fresh):
            recovery.prepare_revision(source, child, revised, self.root / 'new.yaml',
                                      new_binding, rebenchmark=True)
        saved = json.loads((child / 'utensil_selection.json').read_text())
        self.assertEqual(saved['utensil_AP50'], .31)
        self.assertEqual(saved['per_class_AP50'], {'utensil': .31})
        clone = torch.load(child / 'train/weights/last_resume.pt', weights_only=False)
        self.assertEqual(clone['best_fitness'], 0.)
        self.assertNotIn('stopper', clone)
        self.assertEqual(recovery.digest(source), source_hash)
        self.assertEqual(json.loads((child / 'inherited_rebenchmark.json').read_text())['dataset_fingerprint'],
                         'revised-labels')

    def test_rebenchmark_resets_stopping_comparison_but_retains_learning_state(self):
        checkpoint = dict(epoch=14, optimizer={'state': {'step': 180}}, updates=180,
                          ema={'learned': True}, best_fitness=.99,
                          stopper={'best_fitness': .99, 'best_epoch': 2},
                          train_args={'epochs': 80, 'data': 'old.yaml'})
        try:
            clone = recovery.revision_checkpoint(checkpoint, self.root / 'rebased',
                self.root / 'revised.yaml', rebenchmark=True)
        except TypeError as error:
            self.fail('Rebenchmark must discard stale stopping scores: ' + str(error))
        self.assertEqual(clone['epoch'], 14)
        self.assertEqual(clone['optimizer'], checkpoint['optimizer'])
        self.assertEqual(clone['ema'], checkpoint['ema'])
        self.assertEqual(clone['updates'], 180)
        self.assertEqual(clone['best_fitness'], 0.)
        self.assertNotIn('stopper', clone)
        self.assertEqual(checkpoint['best_fitness'], .99)
        self.assertEqual(checkpoint['stopper']['best_fitness'], .99)

    def test_validation_revision_preserves_every_training_label_and_split(self):
        self.assertTrue(hasattr(recovery, 'validate_label_revision'),
                        'Evaluation-label revisions need their own split-safe guard')
        previous = self.build()
        revised = copy.deepcopy(previous)
        next(r for r in revised['records'] if r['split'] == 'val')['rows'].append('3 .8 .8 .1 .1')
        result = recovery.validate_label_revision(previous, revised, 'val')
        self.assertEqual(result['changed_validation_images'], 1)
        self.assertEqual(result['added_objects'], {'utensil': 1})
        next(r for r in revised['records'] if r['split'] == 'train')['rows'].append('3 .8 .8 .1 .1')
        with self.assertRaises(ValueError):
            recovery.validate_label_revision(previous, revised, 'val')

    def test_independent_packet_validation_does_not_split_original_capture_batch(self):
        self.add('val', 'plate__different_meal.jpg', '7 .5 .5 .4 .4', 14)
        try:
            manifest = self.build()
        except ValueError as error:
            self.fail('A separate held-out packet context must be evaluable: ' + str(error))
        self.assertEqual(manifest['counts']['val']['condiment_packet'], 1)
        original = next(r for r in manifest['records'] if r['name'].startswith('condiments__'))
        self.assertEqual(original['split'], 'train')
        bad = copy.deepcopy(manifest['records'])
        next(r for r in bad if r['name'].startswith('condiments__'))['split'] = 'val'
        with self.assertRaises(ValueError):
            recovery.audit(bad)

    def test_training_revision_rejects_validation_changes_and_label_deletions(self):
        self.assertTrue(hasattr(recovery, 'validate_training_revision'),
                        'Need a guard before reusing optimizer state on revised data')
        previous = self.build()
        revised = copy.deepcopy(previous)
        record = next(r for r in revised['records'] if r['split'] == 'train')
        record['rows'].append('3 .8 .8 .1 .1')
        delta = recovery.validate_training_revision(previous, revised)
        self.assertEqual(delta['added_objects'], {'utensil': 1})
        self.assertEqual(delta['changed_training_images'], 1)
        for change in ('validation_label', 'split', 'removed_image', 'deleted_train_label'):
            bad = copy.deepcopy(revised)
            if change == 'validation_label':
                next(r for r in bad['records'] if r['split'] == 'val')['rows'].append('3 .8 .8 .1 .1')
            elif change == 'split':
                next(r for r in bad['records'] if r['split'] == 'val')['split'] = 'train'
            elif change == 'removed_image':
                bad['records'].pop()
            else:
                next(r for r in bad['records'] if r['split'] == 'train')['rows'] = []
            with self.subTest(change=change), self.assertRaises(ValueError):
                recovery.validate_training_revision(previous, bad)

    def test_packet_additions_require_the_same_hash_bound_review_as_utensils(self):
        image, label = self.add('train', 'plate__meal.jpg', '1 .5 .5 .4 .4', 7)
        self.reviews.write_text(json.dumps([dict(name=image.name, image_sha256=recovery.digest(image),
            input_label_sha256=recovery.digest(label), reviewer='test', reason='Visible condiment sachet',
            additions=['7 .8 .8 .1 .1'])]))
        try:
            manifest = self.build()
        except ValueError as error:
            self.fail('Hash-bound packet additions should be accepted: ' + str(error))
        self.assertEqual(manifest['counts']['train']['condiment_packet'], 2)
        self.assertEqual(label.read_text(), '1 .5 .5 .4 .4\n')

    def test_revision_checkpoint_preserves_optimizer_and_eighty_epoch_schedule(self):
        self.assertTrue(hasattr(recovery, 'revision_checkpoint'),
                        'Need a continuation that retains learned optimizer state')
        checkpoint = dict(epoch=7, optimizer={'state': {'step': 116}}, updates=116,
                          ema={'learned': True}, train_args={'epochs': 80, 'data': 'old.yaml',
                          'lr0': .001, 'nbs': 64})
        before = copy.deepcopy(checkpoint)
        clone = recovery.revision_checkpoint(checkpoint, self.root / 'new-run', self.root / 'new.yaml')
        self.assertEqual(checkpoint, before)
        for field in ('epoch', 'optimizer', 'updates', 'ema'):
            self.assertEqual(clone[field], checkpoint[field])
        self.assertEqual(clone['train_args']['epochs'], 80)
        self.assertEqual(clone['train_args']['data'], str((self.root / 'new.yaml').resolve()))
        self.assertEqual(clone['train_args']['lr0'], .001)
        with self.assertRaises(ValueError):
            recovery.revision_checkpoint({**checkpoint, 'optimizer': None}, self.root / 'bad', self.root / 'new.yaml')

    @unittest.skipUnless(ML_AVAILABLE, 'requires ML dependencies; covered by ml-integration')
    def test_resume_options_pass_coverage_factory_before_saved_arguments_load(self):
        from partial_labels import validate_training_options
        from ultralytics.cfg import get_cfg
        options = recovery.resume_options(Path('last_resume.pt'))
        try:
            validate_training_options(get_cfg(overrides=options))
        except ValueError as error:
            self.fail('Resume fails before checkpoint settings can load: ' + str(error))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'source'
        self.output = self.root / 'recovery'
        self.reviews = self.root / 'reviews.json'
        self.reviews.write_text('[]')
        self.names = recovery.class_names()
        self.add('train', 'utensil__WhatsApp-Image-2022-01-27-at-a.jpg', '3 .5 .5 .4 .4', 1)
        self.add('val', 'utensil__WhatsApp-Image-2022-01-20-at-b.jpg', '3 .5 .5 .4 .4', 2)
        self.add('train', 'utensil__stock.jpg', '3 .5 .5 .4 .4', 3)
        self.add('train', 'tartanhacks__orig_IMG_0824_frame_002.jpg', '5 .5 .5 .4 .4', 4)
        self.add('train', 'waste_detection_v3__train__2024-02-05_mov-0001_jpg.jpg', '5 .5 .5 .4 .4', 5)
        self.add('train', 'condiments__train__a.jpg', '7 .5 .5 .4 .4', 6)
        (self.source / 'data.yaml').write_text(yaml.safe_dump(dict(
            path=str(self.source), train='images/train', val='images/val', names=self.names)))

    def add(self, split, name, rows, color):
        image = self.source / 'images' / split / name
        label = self.source / 'labels' / split / Path(name).with_suffix('.txt')
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (24, 24), (color * 25, 10, 30)).save(image)
        label.write_text(rows + '\n')
        return image, label

    def build(self):
        return recovery.build(self.source / 'data.yaml', self.output, self.reviews)

    def test_related_sessions_and_entire_tartan_source_are_held_out(self):
        manifest = self.build()
        by_name = {r['name']: r for r in manifest['records']}
        self.assertEqual(by_name['utensil__WhatsApp-Image-2022-01-27-at-a.jpg']['split'], 'val')
        self.assertEqual(by_name['tartanhacks__orig_IMG_0824_frame_002.jpg']['split'], 'val')
        self.assertEqual(manifest['counts']['val']['napkin'], 1)
        self.assertEqual(manifest['counts']['val']['condiment_packet'], 0)
        self.assertFalse(manifest['independent_test'])
        recovery.verify(self.output)

    def test_original_prefix_is_not_a_split_rule_and_video_frames_stay_together(self):
        self.add('val', 'waste_detection_v3__valid__2024-02-05_mov-0002_jpg.jpg', '5 .5 .5 .4 .4', 8)
        self.add('train', 'dry_wet_waste__a.jpg', '5 .5 .5 .4 .4', 9)
        manifest = self.build()
        selected = [r for r in manifest['records'] if r['name'].startswith('waste_detection_v3__')]
        self.assertEqual({r['split'] for r in selected}, {'val'})

    def test_hash_bound_utensil_addition_preserves_other_classes_and_source(self):
        image, label = self.add('train', 'utensil__tray.jpg', '3 .5 .5 .4 .4\n0 .2 .2 .1 .1', 7)
        before = label.read_bytes()
        self.reviews.write_text(json.dumps([dict(name=image.name, image_sha256=recovery.digest(image),
            input_label_sha256=recovery.digest(label), reviewer='test', reason='Missing spoon',
            additions=['3 .8 .8 .1 .1'])]))
        self.build()
        target = self.output / 'labels/train/utensil__tray.txt'
        self.assertEqual(target.read_text().splitlines(), ['3 .5 .5 .4 .4', '0 .2 .2 .1 .1', '3 .8 .8 .1 .1'])
        self.assertEqual(label.read_bytes(), before)
        self.assertNotEqual(image.stat().st_ino, (self.output / 'images/train' / image.name).stat().st_ino)

    def test_rehashed_label_deletion_still_fails_source_projection(self):
        manifest = self.build()
        name = 'utensil__stock.jpg'
        (self.output / 'labels/train' / Path(name).with_suffix('.txt')).write_text('')
        for record in manifest['records']:
            if record['name'] == name:
                record['rows'] = []
        (self.output / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'projection|provenance'):
            recovery.verify(self.output)

    def test_byte_duplicates_follow_validation_and_are_not_deleted(self):
        source = self.source / 'images/val/utensil__WhatsApp-Image-2022-01-20-at-b.jpg'
        image, label = self.add('train', 'utensil__duplicate.jpg', '3 .5 .5 .4 .4', 10)
        image.write_bytes(source.read_bytes())
        manifest = self.build()
        selected = [r for r in manifest['records'] if r['sha'] == recovery.digest(source)]
        self.assertEqual(len(selected), 2)
        self.assertEqual({r['split'] for r in selected}, {'val'})

    def test_bad_review_hash_and_overwrite_are_rejected(self):
        self.reviews.write_text(json.dumps([dict(name='utensil__stock.jpg', image_sha256='bad',
            input_label_sha256='bad', reviewer='test', reason='test', additions=['3 .8 .8 .1 .1'])]))
        with self.assertRaisesRegex(ValueError, 'hash'):
            self.build()
        self.assertFalse(self.output.exists())
        self.reviews.write_text('[]')
        self.build()
        with self.assertRaises(ValueError):
            self.build()

    def test_training_list_rejects_held_out_paths_and_missing_target_images(self):
        manifest = self.build()
        subset = self.root / 'subset'
        subset.mkdir()
        photos = [self.output / 'images/train' / r['name'] for r in manifest['records'] if r['split'] == 'train']
        listing = subset / 'train.txt'
        listing.write_text('\n'.join(map(str, photos)) + '\n')
        config = subset / 'data.yaml'
        config.write_text(yaml.safe_dump(dict(path=str(self.output), train=str(listing),
            val=str(self.output / 'images/val'), names=self.names)))
        recovery.verify_views(self.output, config, manifest)
        held = next(self.output.joinpath('images/val').iterdir())
        listing.write_text('\n'.join(map(str, photos + [held])) + '\n')
        with self.assertRaisesRegex(ValueError, 'training|held'):
            recovery.verify_views(self.output, config, manifest)
        listing.write_text(str(photos[0]) + '\n')
        with self.assertRaisesRegex(ValueError, 'target'):
            recovery.verify_views(self.output, config, manifest)

    def test_checkpoint_selection_tracks_utensil_and_retains_resume_on_later_decline(self):
        output = self.root / 'model'
        output.mkdir()
        csv = self.root / 'results.csv'
        last = self.root / 'last.pt'
        counts = dict.fromkeys(self.names, 0)
        counts.update(utensil=10, napkin=5)
        callback = recovery.checkpoint_callback(output, counts, minutes=1)
        trainer = SimpleNamespace(epoch=0, csv=csv, last=last, stop=False)
        with patch('utensil_recovery.time.monotonic', side_effect=[0, 30, 65]):
            callback.start(trainer)
            for epoch, scores in enumerate(([.8, .3], [.7, .9])):
                trainer.epoch = epoch
                csv.write_text(f'epoch\n{epoch+1}\n')
                last.write_bytes(f'epoch{epoch+1}'.encode())
                trainer.validator = SimpleNamespace(metrics=SimpleNamespace(
                    nt_per_class=[counts[n] for n in self.names],
                    box=SimpleNamespace(ap_class_index=[3, 5], ap50=scores)))
                callback(trainer)
        self.assertEqual((output / 'train/weights/best_utensil.pt').read_bytes(), b'epoch1')
        self.assertEqual((output / 'train/weights/last_resume.pt').read_bytes(), b'epoch2')
        self.assertTrue(trainer.stop)
        trainer.epoch = 2  # Final revalidation must not bind old metrics to new weights.
        last.write_bytes(b'wrong-final')
        callback(trainer)
        self.assertEqual((output / 'train/weights/best_utensil.pt').read_bytes(), b'epoch1')

    def test_training_hold_preserves_original_and_is_not_a_negative(self):
        image, label = self.add('train', 'dry_wet_waste__napkin.jpg', '5 .5 .5 .4 .4', 12)
        holds = self.root / 'holds.json'
        holds.write_text(json.dumps([dict(name=image.name, image_sha256=recovery.digest(image),
            rows=['5 .5 .5 .4 .4'], reviewer='test', reason='Uncertain target identity; hold entire image')]))
        manifest = recovery.build(self.source / 'data.yaml', self.output, self.reviews, holds)
        held = recovery.training_holds(manifest)
        self.assertEqual(held, {image.name})
        self.assertEqual((self.output / 'labels/train/dry_wet_waste__napkin.txt').read_text(), label.read_text())
        record = next(r for r in manifest['records'] if r['name'] == image.name)
        self.assertTrue(record['rows'])
        views = self.root / 'held-views'
        views.mkdir()
        listing = views / 'train.txt'
        photos = [self.output / 'images/train' / r['name'] for r in manifest['records']
                  if r['split'] == 'train' and r['name'] not in held]
        listing.write_text('\n'.join(map(str, photos)) + '\n')
        config = views / 'data.yaml'
        config.write_text(yaml.safe_dump(dict(path=str(self.output), train=str(listing),
            val=str(self.output / 'images/val'), names=self.names)))
        recovery.verify_views(self.output, config, manifest)
        listing.write_text('\n'.join(map(str, photos + [self.output / 'images/train' / image.name])) + '\n')
        with self.assertRaisesRegex(ValueError, 'held|training'):
            recovery.verify_views(self.output, config, manifest)
        broken = json.loads(holds.read_text())
        broken[0]['image_sha256'] = 'wrong'
        holds.write_text(json.dumps(broken))
        manifest['training_holds']['sha256'] = recovery.digest(holds)
        with self.assertRaisesRegex(ValueError, 'hold'):
            recovery.training_holds(manifest)


if __name__ == '__main__':
    unittest.main()
