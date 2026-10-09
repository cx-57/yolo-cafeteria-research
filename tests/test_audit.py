import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


class AuditTests(unittest.TestCase):
    def run_audit(self, root, output):
        return subprocess.run([sys.executable, str(ROOT / 'src/audit.py'), '--processed', str(root), '--output', str(output), '--config', str(ROOT / 'config.yaml')], capture_output=True, text=True)

    def test_empty_corpus_does_not_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'processed'; root.mkdir()
            run = self.run_audit(root, Path(tmp) / 'report')
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('no_images', (Path(tmp) / 'report/summary.json').read_text())

    def test_detects_duplicate_and_invalid_box_without_mutating_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'processed'
            for source in ['a', 'b']:
                folder = root / source
                (folder / 'images').mkdir(parents=True); (folder / 'labels').mkdir()
                Image.new('RGB', (30, 20), 'orange').save(folder / 'images/a.png')
                (folder / 'labels/a.txt').write_text('0 .95 .5 .2 .2\n' if source == 'a' else '0 .5 .5 .2 .2\n')
            before = (root / 'a/labels/a.txt').read_text()
            run = self.run_audit(root, Path(tmp) / 'report')
            self.assertNotEqual(run.returncode, 0)
            summary = json.loads((Path(tmp) / 'report/summary.json').read_text())
            self.assertEqual(summary['images'], 2)
            self.assertEqual(summary['exact_duplicate_groups'], 1)
            self.assertIn('invalid_box', {i['kind'] for i in summary['issues']})
            self.assertEqual((root / 'a/labels/a.txt').read_text(), before)

    def test_repacked_source_preserves_manifest_split_and_double_underscore_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); root = base / 'processed'; raw = base / 'raw'; manifests = base / 'manifests'
            name = 'utensil__photo'
            for folder in [root / 'utensil', raw / 'utensil/train']:
                (folder / 'images').mkdir(parents=True); (folder / 'labels').mkdir()
                Image.new('RGB', (30, 20), 'orange').save(folder / 'images' / (name+'.png'))
                (folder / 'labels' / (name+'.txt')).write_text(('3' if folder == root / 'utensil' else '0')+' .5 .5 .2 .2\n')
            (raw / 'utensil/data.yaml').write_text('names: [plastic utensil]\n')
            manifests.mkdir(); (manifests / 'utensil.csv').write_text('image,original_split,instances\nutensil__photo.png,valid,1\n')
            command = [sys.executable, str(ROOT / 'src/audit.py'), '--processed', str(root), '--raw', str(raw), '--manifests', str(manifests), '--output', str(base / 'report'), '--config', str(ROOT / 'config.yaml')]
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
            self.assertIn('valid', (base / 'report/inventory.csv').read_text())

    def test_rejects_wrong_raw_image_even_when_filename_and_labels_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); root = base / 'processed'; raw = base / 'raw'
            for folder, color, class_id in [(root / 'utensil', 'blue', 3), (raw / 'utensil/train', 'red', 0)]:
                (folder / 'images').mkdir(parents=True); (folder / 'labels').mkdir()
                Image.new('RGB', (30, 20), color).save(folder / 'images/a.png')
                (folder / 'labels/a.txt').write_text(f'{class_id} .5 .5 .2 .2\n')
            (raw / 'utensil/data.yaml').write_text('names: [plastic utensil]\n')
            run = subprocess.run([sys.executable, str(ROOT / 'src/audit.py'), '--processed', str(root), '--raw', str(raw), '--output', str(base / 'report'), '--config', str(ROOT / 'config.yaml')], capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0, run.stdout+run.stderr)
            summary = json.loads((base / 'report/summary.json').read_text())
            self.assertIn('source_image_mismatch', {i['kind'] for i in summary['issues']})


if __name__ == '__main__':
    unittest.main()
