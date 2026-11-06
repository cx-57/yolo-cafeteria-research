import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import warnings
import zipfile

from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    import v5_dataset as v
except ImportError:
    v = None


def archive_fixture(path, extra=None, corrupt=False):
    rows, payload = [], {}
    for i, split in enumerate(('train', 'val', 'test')):
        buf = io.BytesIO()
        Image.new('RGB', (32, 32), (i * 70, 20, 30)).save(buf, format='PNG')
        ip, lp = f'images/{split}/{i}.png', f'labels/{split}/{i}.txt'
        image, label = buf.getvalue(), b'0 .5 .5 .2 .2\n6 .5 .5 .2 .2\n'
        rows.append(dict(split=split, dataset='fixture', image_path=ip, label_path=lp,
                         image_sha256=hashlib.sha256(image).hexdigest(),
                         label_sha256=hashlib.sha256(label).hexdigest()))
        payload[ip], payload[lp] = image, label
    if corrupt:
        payload[rows[0]['label_path']] = b'changed'
    root = 'v5_original_plus_v4/'
    with zipfile.ZipFile(path, 'w') as z:
        for name, content in payload.items():
            z.writestr(root + name, content)
        z.writestr(root + 'manifest.json', json.dumps(rows))
        z.writestr(root + 'data.yaml', yaml.safe_dump({'names': [
            'food', 'plate', 'utensil', 'hinged_container', 'drink_carton', 'wrapper', 'bottle']}))
        if extra:
            z.writestr(extra, b'bad')
    return rows


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(v, 'six-class preparation module is missing')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_mixed_and_bottle_only_labels(self):
        self.assertEqual(v.strip_bottle('2 .5 .5 .2 .2\n6 .5 .5 .3 .3\n'),
                         ('2 .5 .5 .2 .2\n', 1))
        self.assertEqual(v.strip_bottle('6 .5 .5 .2 .2\n'), ('', 1))

    def test_invalid_class_and_geometry_rejected(self):
        for text in ('7 .5 .5 .2 .2', '2 .1 .5 .5 .2', '2 nan .5 .2 .2',
                     '2.0 .5 .5 .2 .2', '2 .5 .5 0 .2', '6 .5 .5 .2'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                v.strip_bottle(text)

    def test_archive_checks_hashes_and_traversal_before_materialization(self):
        path = self.root / 'source.zip'
        archive_fixture(path, corrupt=True)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            v.read_archive(path, expected_images=3)
        archive_fixture(path, extra='v5_original_plus_v4/../escape.txt')
        with self.assertRaisesRegex(ValueError, 'path'):
            v.read_archive(path, expected_images=3)

    def test_duplicate_archive_members_rejected(self):
        path = self.root / 'source.zip'
        archive_fixture(path)
        with warnings.catch_warnings(), zipfile.ZipFile(path, 'a') as z:
            warnings.simplefilter('ignore', UserWarning)
            z.writestr('v5_original_plus_v4/manifest.json', '[]')
        with self.assertRaisesRegex(ValueError, 'Duplicate|split disagree'):
            v.read_archive(path, expected_images=3)

    def test_manifest_alias_paths_rejected(self):
        path = self.root / 'source.zip'
        rows = archive_fixture(path)
        with zipfile.ZipFile(path) as z:
            payload = {i.filename: z.read(i) for i in z.infolist()}
        rows[1]['image_path'] = rows[0]['image_path']
        payload['v5_original_plus_v4/manifest.json'] = json.dumps(rows).encode()
        with zipfile.ZipFile(path, 'w') as z:
            for name, data in payload.items(): z.writestr(name, data)
        with self.assertRaisesRegex(ValueError, 'Duplicate|split disagree'):
            v.read_archive(path, expected_images=3)

    def test_verified_archive_reports_retained_counts(self):
        path = self.root / 'source.zip'
        archive_fixture(path)
        rows, contents, receipt = v.read_archive(path, expected_images=3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(receipt['removed_bottle_boxes'], {'train': 1, 'val': 1, 'test': 1})
        self.assertEqual(rows[0]['counts'], {0: 1})
        self.assertEqual(contents[rows[0]['image_path']][:8], b'\x89PNG\r\n\x1a\n')

    def test_grouping_keeps_duplicates_and_meals_together_not_entire_source(self):
        rows = [dict(split='train', dataset='school_lunch', group='school_lunch',
                     image_sha256=str(i), counts={c: 1 for c in range(6)}) for i in range(40)]
        rows[0]['image_sha256'] = rows[1]['image_sha256']
        rows[2]['meal_id'] = rows[3]['meal_id'] = 'meal-a'
        rows[0]['split'] = 'val'
        rows[2]['split'] = 'test'
        features = [dict(phash=(1 << (i + 4)) - 1, dhash=1 << i, aspect=2 ** i)
                    for i in range(40)]
        splits, groups = v.assign_splits(rows, features)
        self.assertEqual(splits[0], splits[1])
        self.assertEqual(splits[2], splits[3])
        self.assertNotEqual(splits[0], 'train')
        self.assertNotEqual(splits[2], 'train')
        self.assertEqual(len(set(groups)), 38)
        self.assertEqual(splits.count('val'), 4)
        self.assertEqual(splits.count('test'), 4)

    def test_missing_support_fails(self):
        rows = [dict(split='train', dataset='x', image_sha256=str(i), counts={0: 1})
                for i in range(30)]
        features = [dict(phash=i, dhash=i, aspect=i+1) for i in range(30)]
        with self.assertRaisesRegex(ValueError, 'support'):
            v.assign_splits(rows, features)

    def test_materialization_retains_negative_and_detects_tampering(self):
        path = self.root / 'source.zip'
        archive_fixture(path)
        rows, contents, source = v.read_archive(path, expected_images=3)
        rows[0]['label_text'] = ''
        rows[0]['counts'] = {}
        out = self.root / 'derived'
        v.materialize(rows, contents, source, ['train', 'val', 'test'], [0, 1, 2], out, [])
        receipt = v.verify_dataset(out, require_support=False)
        label = out / receipt['images'][0]['label_path']
        self.assertEqual(label.read_text(), '')
        self.assertTrue((out / receipt['images'][0]['image_path']).is_file())
        label.write_text('6 .5 .5 .2 .2\n')
        with self.assertRaisesRegex(ValueError, 'Changed'):
            v.verify_dataset(out, require_support=False)

    def test_materialization_refuses_existing_output_and_changed_lists(self):
        path = self.root / 'source.zip'
        archive_fixture(path)
        rows, contents, source = v.read_archive(path, expected_images=3)
        out = self.root / 'derived'
        v.materialize(rows, contents, source, ['train', 'val', 'test'], [0, 1, 2], out, [])
        with self.assertRaises(FileExistsError):
            v.materialize(rows, contents, source, ['train', 'val', 'test'], [0, 1, 2], out, [])
        (out / 'train.txt').write_text(str(out / 'images/test/00002.png') + '\n')
        with self.assertRaisesRegex(ValueError, 'Changed'):
            v.verify_dataset(out, require_support=False)

    def test_exclusions_require_unique_exact_source_paths(self):
        rows = [dict(image_path=f'images/train/{i}.png') for i in range(28)]
        self.assertEqual(len(v.validate_exclusions([r['image_path'] for r in rows], rows)), 28)
        for exclusions in ([], ['images/train/0.png'] * 28,
                           [r['image_path'] for r in rows[:-1]] + ['missing.png']):
            with self.assertRaises(ValueError):
                v.validate_exclusions(exclusions, rows)


if __name__ == '__main__': unittest.main()
