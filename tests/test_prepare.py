import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare


class PrepareTests(unittest.TestCase):
    def test_documented_help_works_without_raw_data(self):
        run = subprocess.run([sys.executable, str(ROOT / 'scripts/prepare.py'), '--help'], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('source_key', run.stdout)

    def test_rejects_invalid_geometry_and_nonfinite_coordinates(self):
        for row in ['0 nan .5 .2 .2', '0 .5 .5 0 .2', '0 .95 .5 .2 .2', '0 .5 .5 2 .2', '0 .2 .2 .2 .2 .3 inf']:
            with self.subTest(row=row), self.assertRaises(ValueError):
                prepare.remap_label_line(row, {0: 'apple'}, {'apple': 0}, Path('example.txt'), 1)

    def test_unknown_class_reports_label_location(self):
        with self.assertRaisesRegex(ValueError, r'example.txt:3'):
            prepare.remap_label_line('99 .5 .5 .2 .2', {0: 'apple'}, {'apple': 0}, Path('example.txt'), 3)

    def test_polygon_converts_to_valid_detection_box(self):
        row = prepare.remap_label_line('0 .1 .2 .5 .2 .5 .6', {0: 'apple'}, {'apple': 0}, Path('x.txt'), 1)
        self.assertEqual(int(row.split()[0]), 0)
        for actual, expected in zip(map(float, row.split()[1:]), [.3, .4, .4, .4]):
            self.assertAlmostEqual(actual, expected)

    def test_source_key_is_independent_of_folder_name_and_uses_flat_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'Containers.v7'
            (source / 'train/images').mkdir(parents=True)
            (source / 'train/labels').mkdir()
            (source / 'data.yaml').write_text('names: [Clamshell]\n')
            (source / 'train/images/a.jpg').write_bytes(b'fixture-image')
            (source / 'train/labels/a.txt').write_text('0 .5 .5 .2 .2\n')
            output = Path(tmp) / 'out'
            try:
                count = prepare.process_dataset(source, output, source_key='macworx')
            except TypeError as exc:
                self.fail(str(exc))
            self.assertEqual(count, 1)
            self.assertTrue((output / 'macworx/images/train__a.jpg').exists())
            self.assertEqual((output / 'macworx/labels/train__a.txt').read_text(), '4 .5 .5 .2 .2\n')



class LabelMigrationTests(unittest.TestCase):
    def test_old_class_ids_are_read_using_current_ids(self):
        self.assertEqual(prepare.current_label('5 .5 .5 .2 .2', 'macworx'), '4 .5 .5 .2 .2')
        self.assertEqual(prepare.current_label('8 .5 .5 .2 .2', 'condiments'), '7 .5 .5 .2 .2')
        self.assertEqual(prepare.current_label('4 .5 .5 .2 .2', 'fresh_source'), '4 .5 .5 .2 .2')
        with self.assertRaises(ValueError):
            prepare.current_label('4 .5 .5 .2 .2', 'macworx')


class LvisTests(unittest.TestCase):
    def test_plate_box_and_incomplete_flag(self):
        import lvis
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'train2017').mkdir()
            Image.new('RGB', (100, 60), 'white').save(root / 'train2017/photo.jpg')
            data = {
                'images': [{'id': 10, 'width': 100, 'height': 60,
                            'coco_url': 'http://images.cocodataset.org/train2017/photo.jpg',
                            'not_exhaustive_category_ids': [756]}],
                'annotations': [
                    {'image_id': 10, 'category_id': 756, 'bbox': [10, 10, 30, 20]},
                    {'image_id': 10, 'category_id': 1, 'bbox': [0, 0, 5, 5]},
                ],
            }
            failures, rows = lvis.prepare_split('train', data, root / 'candidates', root)
            self.assertEqual(failures, [])
            self.assertTrue(rows[0][-1])
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                (root / 'candidates/labels/train__10.txt').read_text(),
                '1 0.250000000 0.333333333 0.300000000 0.333333333\n',
            )


if __name__ == '__main__':
    unittest.main()
