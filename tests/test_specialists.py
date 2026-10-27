import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import specialists as s


class SpecialistTests(unittest.TestCase):
    def test_mapping_discards_non_owned_classes_and_keeps_coordinates(self):
        rows = [[1, 2, 30, 40, .8, 0], [2, 3, 4, 5, .9, 1]]
        self.assertEqual(s.map_boxes(rows, ['utensil'], ['utensil']),
                         [[1., 2., 30., 40., .8, 2]])

    def test_wrong_checkpoint_class_names_fail(self):
        with self.assertRaises(ValueError):
            s.map_boxes([], ['glass'], ['bottle'])

    def test_evaluation_requires_complete_coverage_even_for_negative_images(self):
        record = dict(name='a', rows=[], known_classes=list(range(7)))
        with self.assertRaisesRegex(ValueError, 'coverage'):
            s.validate_annotations([record], list(range(8)))
        record['known_classes'] = list(range(8))
        s.validate_annotations([record], list(range(8)))

    def test_invalid_and_duplicate_annotations_fail(self):
        r = dict(name='a', rows=['7 .5 .5 -.2 .1'], known_classes=list(range(8)))
        with self.assertRaises(ValueError):
            s.validate_annotations([r], list(range(8)))
        r['rows'] = []
        with self.assertRaises(ValueError):
            s.validate_annotations([r, r], list(range(8)))

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
