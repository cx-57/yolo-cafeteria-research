import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
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
