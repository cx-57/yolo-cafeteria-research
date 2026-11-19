import json
import sys
import unittest
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import replay_sorting as r
import specialists as s
import disposal as d


class ReplaySortingTests(unittest.TestCase):
    def setUp(self):
        self.config = s.read(s.CONFIG)
        self.policy = d.validate_policy(s.read(d.DEFAULT_RULES))
        self.config_hash = 'frozen-config-sha'
        model_settings = {model['id']: s.inference_settings(self.config, model)
                          for model in self.config['models']}
        self.receipt = dict(
            names=s.NAMES, settings=s.inference_settings(self.config),
            config_sha256=self.config_hash, models=self.config['models'],
            model_settings=model_settings,
            box_format='normalized_xyxy_conf_class',
            images=[dict(name='a.jpg', sha256='image-a'),
                    dict(name='b.jpg', sha256='image-b')],
            predictions={
                'a.jpg': [
                    [0, 0, .2, .2, .91, 0],
                    [.2, .2, .4, .4, .81, 6],
                    [.4, .4, .6, .6, .78, 4],
                    [.6, .6, .8, .8, .90, 1],
                    [.8, .8, .95, .95, .40, 6],
                ],
                'b.jpg': [],
            },
        )

    def test_replay_counts_no_empty_requirement(self):
        result = r.replay(self.receipt, self.policy, self.config,
                          self.config_hash, expected_images=2)
        self.assertEqual(result['category_counts']['compost'], 1)
        self.assertEqual(result['category_counts']['recycling'], 2)
        self.assertEqual(result['category_counts']['review'], 2)
        self.assertEqual(result['category_counts']['landfill'], 0)
        self.assertEqual(result['detections'], 5)
        self.assertEqual(result['images'], 2)
        self.assertIsNone(result['actual_disposal_accuracy'])

    def test_nested_evaluation_receipt_supported(self):
        nested = r.replay({'prediction_receipt': self.receipt},
                          self.policy, self.config, self.config_hash)
        self.assertEqual(nested['category_counts']['recycling'], 2)

    def test_image_verification_uses_manifest_paths_and_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'original.jpg'
            path.write_bytes(b'original image bytes')
            receipt = dict(images=[dict(name='original.jpg', sha256=s.sha(path))])
            manifest = dict(records=[dict(name='original.jpg', sha256=s.sha(path),
                                          image_path=str(path))])
            r.verify_replay_images(receipt, manifest)
            path.write_bytes(b'changed image bytes')
            with self.assertRaisesRegex(ValueError, 'Changed image'):
                r.verify_replay_images(receipt, manifest)

    def test_image_verification_rejects_wrong_manifest_cohort(self):
        manifest = dict(records=[dict(name='other.jpg', sha256='image-a',
                                      image_path='other.jpg')])
        with self.assertRaisesRegex(ValueError, 'cohort'):
            r.verify_replay_images(self.receipt, manifest)

    def test_mismatched_or_truncated_receipts_rejected(self):
        with self.assertRaisesRegex(ValueError, 'cohort'):
            r.replay(self.receipt, self.policy, self.config,
                     self.config_hash, expected_images=248)
        changed = json.loads(json.dumps(self.receipt))
        changed['config_sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            r.replay(changed, self.policy, self.config, self.config_hash)
        changed = json.loads(json.dumps(self.receipt))
        del changed['predictions']['b.jpg']
        with self.assertRaises(ValueError):
            r.replay(changed, self.policy, self.config, self.config_hash)
        changed = json.loads(json.dumps(self.receipt))
        changed['predictions']['a.jpg'][0][4] = 1.5
        with self.assertRaises(ValueError):
            r.replay(changed, self.policy, self.config, self.config_hash)


if __name__ == '__main__':
    unittest.main()
