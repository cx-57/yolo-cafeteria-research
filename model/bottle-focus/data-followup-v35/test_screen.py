"""Real image controls for staging overlap screening, without model inference."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image

SPEC = importlib.util.spec_from_file_location('followup_screen', Path(__file__).with_name('screen_followup.py'))
screen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(screen)

class OverlapControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        pixels = np.random.default_rng(43).integers(0, 256, (500, 640, 3), dtype=np.uint8)
        self.image = Image.fromarray(pixels)
        self.original = self.path/'original.png'
        self.image.save(self.original)

    def test_pixel_alias_with_changed_container_is_held(self):
        alias = self.path/'alias.bmp'
        self.image.save(alias)
        a, b = screen.features(self.original), screen.features(alias)
        result = screen.compare(a, [('monitor/protected', b)])
        self.assertTrue(result['hold'])
        self.assertEqual(result['pixel_matches'], ['monitor/protected'])

    def test_reencoded_crop_of_benchmark_is_held_by_local_features(self):
        alias = self.path/'crop.jpg'
        self.image.crop((40, 30, 600, 470)).save(alias, quality=96)
        result = screen.compare(screen.features(alias), [('benchmark/protected', screen.features(self.original))])
        self.assertTrue(result['hold'])
        self.assertTrue(result['feature_matches'])

    def test_identical_file_is_held_even_without_sift_features(self):
        blank = self.path/'blank.png'
        Image.new('RGB', (64,64), 'white').save(blank)
        result = screen.compare(screen.features(blank), [('monitor/blank', screen.features(blank))])
        self.assertTrue(result['hold'])
        self.assertEqual(result['byte_matches'], ['monitor/blank'])

    def test_replaced_protected_photo_rejected_before_screening(self):
        with self.assertRaisesRegex(ValueError, 'Protected image changed'):
            screen.protected_features(self.original, '0'*64)

    def test_review_input_changed_after_loading_rejected(self):
        path = self.path/'review.json'
        path.write_text('{"accepted": [1]}')
        value, digest = screen.snapshot(path)
        self.assertEqual(value['accepted'], [1])
        path.write_text('{"accepted": [2]}')
        with self.assertRaisesRegex(ValueError, 'Input changed'):
            screen.check_inputs({str(path): digest})

if __name__ == '__main__':
    unittest.main()
