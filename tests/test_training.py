import copy
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image, ImageDraw
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
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
