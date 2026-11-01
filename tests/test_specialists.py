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
