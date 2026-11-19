"""Model scoring, data integrity, training, and evaluation tests."""

# --- test_specialists.py ---
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import specialists as s


class SpecialistTests(unittest.TestCase):
    def test_threshold_metrics_penalize_duplicates_misses_and_negative_images(self):
        records = [dict(name='a', known_classes=list(range(7)), rows=[
            '0 0.2 0.2 0.2 0.2', '6 0.7 0.7 0.2 0.2']),
            dict(name='empty', known_classes=list(range(7)), rows=[])]
        predictions = {'a': [[.1, .1, .3, .3, .9, 0],
                              [.1, .1, .3, .3, .8, 0],
                              [.6, .6, .8, .8, .49, 6]],
                       'empty': [[.1, .1, .3, .3, .5, 6]]}
        result = s.recognition_at_threshold(records, predictions)
        self.assertEqual(result['per_class']['food']['tp'], 1)
        self.assertEqual(result['per_class']['food']['fp'], 1)
        self.assertEqual(result['per_class']['bottle']['fn'], 1)
        self.assertEqual(result['per_class']['bottle']['fp'], 1)
        self.assertAlmostEqual(result['micro']['precision'], 1/3)
        self.assertAlmostEqual(result['micro']['recall'], .5)
        self.assertAlmostEqual(result['micro']['f1'], .4)

    def test_threshold_metrics_do_not_count_wrong_class_as_recognition(self):
        records = [dict(name='a', known_classes=list(range(7)),
                        rows=['0 0.2 0.2 0.2 0.2'])]
        result = s.recognition_at_threshold(
            records, {'a': [[.1, .1, .3, .3, .9, 6]]})
        self.assertEqual(result['micro']['tp'], 0)
        self.assertEqual(result['micro']['fp'], 1)
        self.assertEqual(result['micro']['fn'], 1)

    def test_model_resolution_overrides_leave_scoring_settings_and_other_models_unchanged(self):
        config = dict(models=[], settings=dict(s.SETTINGS))
        higher = dict(id='higher', settings=dict(imgsz=800))
        ordinary = dict(id='ordinary')
        resolved = s.inference_settings(config, higher)
        self.assertEqual(resolved['imgsz'], 800)
        self.assertEqual({k: v for k, v in resolved.items() if k != 'imgsz'},
                         {k: v for k, v in s.SETTINGS.items() if k != 'imgsz'})
        self.assertEqual(s.inference_settings(config, ordinary), s.SETTINGS)
        self.assertEqual(config['settings'], s.SETTINGS)

    def test_per_model_overrides_cannot_change_thresholds_or_silently_round_size(self):
        for override in [dict(conf=.1), dict(iou=.5), dict(augment=True),
                         dict(imgsz=799), dict(imgsz=0), dict(imgsz=True)]:
            with self.assertRaises(ValueError):
                s.inference_settings(dict(models=[]), dict(settings=override))

    def test_mixed_resolution_receipts_require_explicit_correct_applied_profiles(self):
        model = dict(id='higher', sha256='weights', settings=dict(imgsz=800))
        config = dict(models=[model])
        record = dict(name='a', sha256='image')
        receipt = dict(names=s.NAMES, settings=s.SETTINGS, config_sha256='config',
                       models=[model], box_format='normalized_xyxy_conf_class',
                       images=[record], predictions={'a': []})
        with self.assertRaises(ValueError):
            s.validate_receipt(receipt, [record], config, 'config')
        receipt['model_settings'] = {'higher': s.SETTINGS}
        with self.assertRaises(ValueError):
            s.validate_receipt(receipt, [record], config, 'config')
        receipt['model_settings'] = {'higher': dict(s.SETTINGS, imgsz=800)}
        self.assertEqual(s.validate_receipt(receipt, [record], config, 'config')['model_settings'],
                         receipt['model_settings'])

    def test_active_pipeline_has_four_models_and_seven_stable_class_ids(self):
        self.assertEqual(s.NAMES, ['food', 'plate', 'utensil', 'hinged_container',
                                  'drink_carton', 'wrapper', 'bottle'])
        models = s.read(s.CONFIG)['models']
        self.assertEqual(len(models), 4)
        self.assertEqual(sorted(c for m in models for c in m['classes']), sorted(s.NAMES))

    def test_retired_packet_class_cannot_appear_in_predictions(self):
        with self.assertRaises(ValueError):
            s.validate_prediction_rows([[.1,.2,.3,.4,.9,7]])

    def test_mapping_discards_non_owned_classes_and_keeps_coordinates(self):
        rows = [[1, 2, 30, 40, .8, 0], [2, 3, 4, 5, .9, 1]]
        self.assertEqual(s.map_boxes(rows, ['utensil'], ['utensil']),
                         [[1., 2., 30., 40., .8, 2]])

    def test_wrong_checkpoint_class_names_fail(self):
        with self.assertRaises(ValueError):
            s.map_boxes([], ['glass'], ['bottle'])

    def test_evaluation_requires_complete_coverage_even_for_negative_images(self):
        record = dict(name='a', rows=[], known_classes=list(range(6)))
        with self.assertRaisesRegex(ValueError, 'coverage'):
            s.validate_annotations([record], list(range(7)))
        record['known_classes'] = list(range(7))
        s.validate_annotations([record], list(range(7)))

    def test_invalid_and_duplicate_annotations_fail(self):
        r = dict(name='a', rows=['6 .5 .5 -.2 .1'], known_classes=list(range(7)))
        with self.assertRaises(ValueError):
            s.validate_annotations([r], list(range(7)))
        r['rows'] = []
        with self.assertRaises(ValueError):
            s.validate_annotations([r, r], list(range(7)))

    def test_perfect_predictions_and_negative_false_positive(self):
        records = [dict(name='a', rows=['0 .5 .5 .2 .2']), dict(name='b', rows=[])]
        good = {'a': [[.4, .4, .6, .6, .8, 0]], 'b': []}
        score = s.score_predictions(records, good, [0])
        self.assertGreater(score['AP50'], .99)
        good['b'] = [[.4, .4, .6, .6, .99, 0]]
        self.assertLess(s.score_predictions(records, good, [0])['AP50'], .6)

    def test_missing_class_support_and_missing_prediction_fail(self):
        records = [dict(name='a', rows=['0 .5 .5 .2 .2'])]
        with self.assertRaisesRegex(ValueError, 'support'):
            s.score_predictions(records, {'a': []}, [0, 1])
        with self.assertRaisesRegex(ValueError, 'prediction'):
            s.score_predictions(records, {}, [0])

    def test_frozen_predictions_round_trip_by_image_hash_and_evaluation_envelope(self):
        record = dict(name='a.jpg', sha256='image')
        config = dict(models=[dict(id='one', sha256='weights')])
        receipt = dict(names=s.NAMES, settings=s.SETTINGS, config_sha256='config',
                       models=config['models'], box_format='normalized_xyxy_conf_class',
                       images=[dict(name='/tmp/a.jpg', sha256='image')],
                       predictions={'/tmp/a.jpg': [[.1, .2, .3, .4, .9, 0]]})
        result = s.validate_receipt(dict(prediction_receipt=receipt), [record], config, 'config')
        self.assertEqual(result['predictions']['a.jpg'], receipt['predictions']['/tmp/a.jpg'])
        self.assertEqual(s.validate_receipt(receipt, [record], config, 'config'), result)
        for field, value in [('models', []), ('box_format', 'xywh')]:
            with self.assertRaises(ValueError):
                s.validate_receipt(dict(receipt, **{field: value}), [record], config, 'config')

    def test_malformed_frozen_prediction_rows_are_rejected(self):
        for row in [[.1,.2,.3,.4,.9,.5], [.3,.2,.1,.4,.9,0],
                    [.1,.2,.3,.4,float('nan'),0], [.1,.2,.3,.4,2,0]]:
            with self.assertRaises(ValueError):
                s.validate_prediction_rows([row])

    def test_any_specialists_training_background_excludes_shared_image(self):
        records = [dict(name='a', sha256='a', pixels='p'), dict(name='b', sha256='b', pixels='q')]
        training = [dict(name='different', sha='other', pixels='p', split='train', rows=[])]
        self.assertEqual(s.training_collisions(records, training), {'a'})
        training[0]['split'] = 'val'
        self.assertEqual(s.training_collisions(records, training), set())

    def test_resolution_profile_is_uniform_and_receipts_cannot_cross_profiles(self):
        config = dict(models=[], settings=dict(s.SETTINGS, imgsz=800))
        self.assertEqual(s.inference_settings(config)['imgsz'], 800)
        receipt = dict(names=s.NAMES, settings=s.SETTINGS, config_sha256='config', models=[],
                       box_format='normalized_xyxy_conf_class', images=[], predictions={})
        with self.assertRaises(ValueError):
            s.validate_receipt(receipt, [], config, 'config')
        receipt['settings'] = config['settings']
        self.assertEqual(s.validate_receipt(receipt, [], config, 'config')['settings']['imgsz'], 800)
        self.assertEqual(s.SETTINGS['imgsz'], 640)


# --- test_storage.py ---
import tempfile
import storage


class StorageTests(unittest.TestCase):
    def test_external_dataset_resolution_preserves_legacy_image_records(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'repo'
            data = Path(temp) / 'datasets'
            image = data / 'processed/source/images/a.jpg'
            image.parent.mkdir(parents=True)
            image.write_bytes(b'original image')
            legacy = root / 'data/processed/source/images/a.jpg'
            self.assertEqual(storage.resolve_image(legacy, root=root, data=data), image)
            self.assertEqual(storage.resolve_image(image, root=root, data=data).read_bytes(), b'original image')

    def test_missing_source_is_reported_instead_of_guessed(self):
        with self.assertRaises(FileNotFoundError):
            storage.resolve_image(Path('/nonexistent/source/image.jpg'))

    def test_default_dataset_storage_is_outside_the_repository(self):
        self.assertFalse(storage.DATA.is_relative_to(storage.ROOT))


# --- test_training.py ---
import copy
import shutil
from PIL import Image, ImageDraw
import yaml

import training as t


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for i, split in enumerate(['train', 'val', 'test']):
            images = self.root / 'images' / split
            labels = self.root / 'labels' / split
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            im = Image.new('RGB', (100, 100), (10 + i * 10, 10, 10))
            ImageDraw.Draw(im).rectangle((35, 35, 64, 64), fill='white')
            im.save(images / 'a.png')
            (labels / 'a.txt').write_text('0 .5 .5 .3 .3\n')
        self.yaml = self.root / 'data.yaml'
        self.yaml.write_text(yaml.safe_dump(dict(path=str(self.root), names=t.NAMES,
                                                train='images/train', val='images/val', test='images/test')))

    def test_profile_is_moderate_and_does_not_enable_composite_cuts(self):
        profile = t.load_profile()
        self.assertEqual(profile['augmentation']['degrees'], 15)
        self.assertEqual(profile['augmentation']['flipud'], 0)
        for key in ['mosaic', 'mixup', 'cutmix', 'copy_paste']:
            self.assertEqual(profile['augmentation'][key], 0)
        self.assertLessEqual(profile['augmentation']['perspective'], .0005)

    def test_inspection_detects_identical_photos_across_splits(self):
        shutil.copyfile(self.root/'images/train/a.png', self.root/'images/val/a.png')
        with self.assertRaisesRegex(ValueError, 'overlap'):
            t.inspect_dataset(self.yaml, require_all_classes=False)

    def test_seven_class_mapping_and_explicit_labels_are_required(self):
        spec = yaml.safe_load(self.yaml.read_text())
        spec['names'] = list(reversed(t.NAMES))
        self.yaml.write_text(yaml.safe_dump(spec))
        with self.assertRaisesRegex(ValueError, 'class'):
            t.inspect_dataset(self.yaml, require_all_classes=False)
        spec['names'] = t.NAMES
        spec['nc'] = 8
        self.yaml.write_text(yaml.safe_dump(spec))
        with self.assertRaisesRegex(ValueError, 'class'):
            t.inspect_dataset(self.yaml, require_all_classes=False)
        spec['nc'] = 7
        self.yaml.write_text(yaml.safe_dump(spec))
        (self.root/'labels/train/a.txt').unlink()
        with self.assertRaisesRegex(ValueError, 'label'):
            t.inspect_dataset(self.yaml, require_all_classes=False)

    def test_cut_off_food_and_retired_class_labels_are_rejected(self):
        label = self.root/'labels/train/a.txt'
        for row in ['0 .1 .5 .2 .3\n', '7 .5 .5 .3 .3\n']:
            label.write_text(row)
            with self.assertRaises(ValueError):
                t.inspect_dataset(self.yaml, require_all_classes=False)

    def test_regular_training_requires_all_seven_classes_but_smoke_can_inspect(self):
        info = t.inspect_dataset(self.yaml, require_all_classes=False)
        self.assertEqual(info['missing_classes']['train'], t.NAMES[1:])
        with self.assertRaisesRegex(ValueError, 'support'):
            t.inspect_dataset(self.yaml, require_all_classes=True)

    def test_reviewed_training_crops_do_not_relax_validation_or_test(self):
        spec = yaml.safe_load(self.yaml.read_text())
        spec['allow_cropped_food_train'] = True
        self.yaml.write_text(yaml.safe_dump(spec))
        (self.root/'labels/train/a.txt').write_text('0 .1 .5 .2 .3\n')
        t.inspect_dataset(self.yaml, require_all_classes=False)
        for split in ['val', 'test']:
            label = self.root/f'labels/{split}/a.txt'
            label.write_text('0 .1 .5 .2 .3\n')
            with self.assertRaisesRegex(ValueError, 'Cut-off'):
                t.inspect_dataset(self.yaml, require_all_classes=False)
            label.write_text('0 .5 .5 .3 .3\n')

    def test_crop_permission_is_boolean_and_never_allows_out_of_bounds_boxes(self):
        spec = yaml.safe_load(self.yaml.read_text())
        spec['allow_cropped_food_train'] = 'true'
        self.yaml.write_text(yaml.safe_dump(spec))
        with self.assertRaisesRegex(ValueError, 'boolean'):
            t.inspect_dataset(self.yaml, require_all_classes=False)
        spec['allow_cropped_food_train'] = True
        self.yaml.write_text(yaml.safe_dump(spec))
        (self.root/'labels/train/a.txt').write_text('0 .05 .5 .2 .3\n')
        with self.assertRaisesRegex(ValueError, 'Invalid class/box'):
            t.inspect_dataset(self.yaml, require_all_classes=False)

    def test_profile_cannot_resume_another_run_or_change_class_scope(self):
        for key, value in [('resume', True), ('single_cls', True), ('classes', [0]),
                           ('fraction', .5), ('save_dir', '/tmp/other-run')]:
            profile = t.load_profile()
            profile['training'][key] = value
            path = self.root / 'profile.yaml'
            path.write_text(yaml.safe_dump(profile))
            with self.assertRaisesRegex(ValueError, 'profile'):
                t.load_profile(path)

    def test_frozen_copies_use_audited_list_not_native_relative_path_rules(self):
        folder = self.root/'lists'
        folder.mkdir()
        listing = folder/'train.txt'
        listing.write_text('../images/train/a.png\n')
        spec = yaml.safe_load(self.yaml.read_text())
        spec['train'] = str(listing)
        self.yaml.write_text(yaml.safe_dump(spec))
        info = t.inspect_dataset(self.yaml, require_all_classes=False)
        Image.new('RGB', (100,100), 'red').save(self.root/'images/train/late.png')
        safe = t.snapshot_dataset(info, self.root/'snapshot')
        native = Path(safe['native_data']['train'])
        self.assertEqual(len(list(native.glob('*'))), 1)
        self.assertFalse(next(native.glob('*')).is_symlink())
        self.assertEqual(t.sha(next(native.glob('*'))), info['records']['train'][0]['sha256'])

    def test_repairable_jpeg_is_rejected_without_changing_original(self):
        image = self.root/'images/train/bad.jpg'
        Image.new('RGB',(100,100),'red').save(image)
        broken = image.read_bytes()[:-2]
        image.write_bytes(broken)
        (self.root/'labels/train/bad.txt').write_text('0 .5 .5 .3 .3\n')
        with self.assertRaisesRegex(ValueError, 'image|JPEG|corrupt'):
            t.inspect_dataset(self.yaml, require_all_classes=False)
        self.assertEqual(image.read_bytes(), broken)

    def test_native_train_augmentation_moves_box_with_object_and_val_is_deterministic(self):
        import random
        import numpy as np
        import torch
        info = t.inspect_dataset(self.yaml, require_all_classes=False)
        profile = t.load_profile()
        profile['training']['imgsz'] = 160
        before = copy.deepcopy(profile)
        train = t.native_dataset(info, 'train', profile)
        val = t.native_dataset(info, 'val', profile)
        random.seed(11); np.random.seed(11)
        augmented = train[0]
        self.assertEqual(len(augmented['bboxes']), 1)
        x, y, w, h = augmented['bboxes'][0].tolist()
        pixels = augmented['img'].float().mean(0) > 180
        ys, xs = torch.where(pixels)
        self.assertGreater(len(xs), 20)
        self.assertLessEqual((x-w/2)*160, float(xs.min())+3)
        self.assertGreaterEqual((x+w/2)*160, float(xs.max())-3)
        self.assertLessEqual((y-h/2)*160, float(ys.min())+3)
        self.assertGreaterEqual((y+h/2)*160, float(ys.max())-3)
        first = val[0]
        random.seed(99); np.random.seed(99)
        second = val[0]
        self.assertTrue(torch.equal(first['img'], second['img']))
        self.assertTrue(torch.equal(first['bboxes'], second['bboxes']))
        self.assertFalse(torch.equal(augmented['img'], first['img']))
        self.assertEqual(profile, before)


# --- test_precision_sweep.py ---
"""Regression tests: 97% micro-precision goal cannot be claimed by cherry-picking test."""
import json
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import precision_sweep as p

class PrecisionSweepTests(unittest.TestCase):
    def test_threshold_grid_contains_fixed_baseline_and_high_confidence(self):
        seq=p.thresholds()
        self.assertEqual(.5,seq[0])
        self.assertIn(.97,seq)
        self.assertEqual(.995,seq[-1])
        self.assertEqual(len(seq),len(set(seq)))

    def test_choose_best_recall_with_precision_floor(self):
        rows=[
            dict(threshold=.50,precision=.9342,recall=.7835,predictions=775),
            dict(threshold=.63,precision=.969,recall=.77,predictions=730),
            dict(threshold=.70,precision=.975,recall=.735,predictions=660),
            dict(threshold=.80,precision=.99,recall=.67,predictions=610),
            dict(threshold=.99,precision=1.,recall=.001,predictions=1),
        ]
        chosen=p.select_threshold(rows)
        self.assertEqual(chosen['threshold'],.70)

    def test_no_spurious_97_with_vanishing_coverage(self):
        rows=[
            dict(threshold=.50,precision=.93,recall=.82,predictions=1000),
            dict(threshold=.99,precision=1.0,recall=.02,predictions=10),
        ]
        self.assertIsNone(p.select_threshold(rows))
        self.assertIsNone(p.select_threshold(rows,min_recall=.00,min_predictions=100))

    def test_target_must_be_finite_real(self):
        for target in [0,-1,1.1,float('nan')]:
            if target!=target: # NaN
                # ensure NaN is rejected, not silently interpreted as no threshold
                with self.assertRaises(ValueError):
                    p.select_threshold([],target=target)
            else:
                with self.assertRaises(ValueError):
                    p.select_threshold([],target=target)

    def test_test_results_do_not_select_threshold(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'result.json'
            config={'models':[]}
            freeze={'protected_sha256':{},'manifests':{}}
            manifest=Path(td)/'validation.json'
            test=Path(td)/'test.json'
            manifest.write_text('val');test.write_text('test')
            records=[{'name':'a'}]
            predictions={'a':[]}
            def mock_summary(_records,_preds,threshold):
                if _records[0]['name']=='validation':
                    return dict(threshold=threshold,precision=.93 if threshold<.75 else .98,
                                recall=.82 if threshold<.75 else .64,predictions=200,
                                tp=0,fp=0,fn=0,objects=0,f1=None)
                else:
                    return dict(threshold=threshold,precision=.5 if threshold<.95 else 1.0,
                                recall=.7,predictions=200,tp=0,fp=0,fn=0,objects=0,f1=None)
            with patch.object(p.s,'read',side_effect=[freeze,config]),\
                 patch.object(p.research_audit,'verify_freeze'),\
                 patch.object(p.s,'sha',return_value='fake'),\
                 patch.object(p,'load_cohort',side_effect=[
                     ([{'name':'validation'}],predictions,'valsha'),
                     ([{'name':'test'}],predictions,'testsha')]),\
                 patch.object(p,'summarize',side_effect=mock_summary):
                result=p.run(manifest,test,None,None,out,'cpu',.97,.60,100,False)
            self.assertEqual(.75,result['test']['threshold_selected_without_test'])
            self.assertEqual(.5,result['test']['fixed_threshold_metrics']['precision'])
            self.assertFalse(result['test']['test_precision_at_least_target'])
            self.assertTrue(out.exists())
            self.assertFalse(result['independent_test'])

if __name__=='__main__':unittest.main()


# --- test_precision_per_class.py ---
"""Classwise confidence optimizer must not hide recall failures or modify models."""

import precision_per_class as q

class PrecisionPerClassTests(unittest.TestCase):
    def make_rows(self):
        result=[]
        for t in (.50,.70):
            per={}
            for i,name in enumerate(q.CLASS_NAMES):
                if i==0:
                    tp,fp=(30,15) if t==.5 else (25,2)
                else:
                    tp,fp=(30,0) if t==.5 else (20,0)
                per[name]=dict(tp=tp,fp=fp,fn=40-tp,objects=40,predictions=tp+fp,
                               precision=tp/(tp+fp),recall=tp/40)
            result.append(dict(threshold=t,per_class=per))
        return result

    def test_classwise_combination_can_improve_precision_without_wholesale_recall_loss(self):
        chosen=q.optimize(self.make_rows(),goal=.97,min_recall=.60,min_predictions=100)['selected']
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen['tp'],205)
        self.assertEqual(chosen['fp'],2)
        self.assertGreater(chosen['precision'],.97)
        self.assertGreater(chosen['recall'],.70)
        self.assertEqual(chosen['thresholds']['food'],.70)
        self.assertEqual(chosen['thresholds']['plate'],.50)

    def test_recalls_and_prediction_guards_cannot_be_bypassed(self):
        self.assertIsNone(q.optimize(self.make_rows(),min_recall=.95)['selected'])
        self.assertIsNone(q.optimize(self.make_rows(),min_predictions=300)['selected'])

    def test_no_97_at_any_threshold_returns_no_qualified_solution(self):
        rows=self.make_rows()
        for r in rows:
            for name,metrics in r['per_class'].items():
                metrics['fp']+=20
        self.assertIsNone(q.optimize(rows,goal=.97)['selected'])

    def test_per_class_filter_applies_separate_cutoffs(self):
        names=q.CLASS_NAMES
        rows=['0 .2 .2 .2 .2','1 .7 .7 .2 .2']
        records=[dict(name='a',known_classes=list(range(7)),rows=rows)]
        pred={'a':[[.1,.1,.3,.3,.7,0],[.6,.6,.8,.8,.6,1]]}
        thresholds=dict.fromkeys(names,.5)
        baseline=q.evaluate_thresholds(records,pred,thresholds)
        self.assertEqual((baseline['tp'],baseline['fp']), (2,0))
        thresholds['food']=.8
        changed=q.evaluate_thresholds(records,pred,thresholds)
        self.assertEqual((changed['tp'],changed['fn']), (1,1))

    def test_invalid_threshold_vectors_rejected(self):
        with self.assertRaises(ValueError):
            q.evaluate_thresholds([],{},dict(food=.6))
        with self.assertRaises(ValueError):
            q.optimize(self.make_rows(),goal=float('nan'))


# --- test_sorting_evaluation.py ---
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_sorting as e


def obj(box, category):
    return dict(box_xyxy_normalized=box, category=category)


def pred(box, category, confidence):
    return dict(box_xyxy_normalized=box, category=category,
                detection_confidence=confidence)


class SortingEvaluationTests(unittest.TestCase):
    def test_correct_wrong_review_missed_and_extra_are_distinct(self):
        boxes = [
            [0, 0, .2, .2],
            [.25, 0, .45, .2],
            [.5, 0, .7, .2],
            [.75, 0, .95, .2],
        ]
        truth = {"images": {"a.jpg": [
            obj(boxes[0], "compost"), obj(boxes[1], "recycling"),
            obj(boxes[2], "landfill"), obj(boxes[3], "compost")]}}
        choices = {"decisions": {"a.jpg": [
            pred(boxes[0], "compost", .95),
            pred(boxes[1], "landfill", .9),
            pred(boxes[2], "review", .8),
            pred([0, .5, .2, .7], "recycling", .7)]}}
        result = e.evaluate(choices, truth)
        self.assertEqual(result["correct_automatic"], 1)
        self.assertEqual(result["wrong_bin_automatic"], 1)
        self.assertEqual(result["matched_review"], 1)
        self.assertEqual(result["missed_objects"], 1)
        self.assertEqual(result["unmatched_predictions"]["recycling"], 1)
        self.assertAlmostEqual(result["automatic_coverage_of_objects"], .5)
        self.assertAlmostEqual(result["accuracy_among_matched_automatic_decisions"], .5)
        self.assertAlmostEqual(result["correct_automatic_over_all_labeled_objects"], .25)
        self.assertAlmostEqual(result["precision_including_unmatched_auto_predictions"], 1/3)
        self.assertEqual(result["confusion_by_true_bin"]["recycling"]["landfill"], 1)

    def test_does_not_match_twice_and_counts_empty_image_false_positive(self):
        b = [.1, .1, .5, .5]
        truth = {"images": {"a.jpg": [obj(b, "compost")], "b.jpg": []}}
        choices = {"decisions": {"a.jpg": [
            pred(b, "compost", .9), pred(b, "compost", .8)],
            "b.jpg": [pred(b, "landfill", .7)]}}
        result = e.evaluate(choices, truth)
        self.assertEqual(result["correct_automatic"], 1)
        self.assertEqual(result["unmatched_predictions"]["compost"], 1)
        self.assertEqual(result["unmatched_predictions"]["landfill"], 1)
        self.assertAlmostEqual(result["precision_including_unmatched_auto_predictions"], 1/3)

    def test_images_cannot_be_dropped(self):
        truth = {"images": {"a.jpg": [], "b.jpg": []}}
        with self.assertRaisesRegex(ValueError, "match"):
            e.evaluate({"decisions": {"a.jpg": []}}, truth)

    def test_invalid_truth_or_predictions_rejected(self):
        b = [.1, .1, .5, .5]
        for category in ("review", "unknown"):
            with self.assertRaises(ValueError):
                e.evaluate({"decisions": {"x": []}},
                           {"images": {"x": [obj(b, category)]}})
        with self.assertRaises(ValueError):
            e.evaluate({"decisions": {"x": [pred(b, "compost", float("nan"))]}},
                       {"images": {"x": [obj(b, "compost")]}})
        with self.assertRaises(ValueError):
            e.evaluate({"decisions": {"x": []}},
                       {"images": {"x": [obj([0, 0, 1.1, 1], "compost")]}})
