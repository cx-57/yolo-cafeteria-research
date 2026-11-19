"""Exploratory runner tests. No real YOLO checkpoints or independent test labels."""
import hashlib
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import explore_candidates as ex


def make_zip(path, changed_hash=False, missing=False, duplicate=False):
    image=Image.new('RGB',(50,40),'orange')
    bio=io.BytesIO();image.save(bio,format='JPEG');data=bio.getvalue()
    sha=hashlib.sha256(data).hexdigest()
    row=dict(path='images/test.jpg',sha256='0'*64 if changed_hash else sha,
             source_hint='bottle',source='synthetic')
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('manifest.json',json.dumps({'independent_certified_images':0,
                   'images':[row]+([row] if duplicate else [])}))
        if not missing:z.writestr('images/test.jpg',data)


class ExploratoryWorkflowTests(unittest.TestCase):
    def test_real_bundle_verifies_without_inference_when_present(self):
        actual=Path('/mnt/data/cafeteria-44-image-review-kit-final.zip')
        if not actual.exists():self.skipTest('Conversation bundle not in CI')
        with tempfile.TemporaryDirectory() as td:
            manifest=ex.unpack_checked(actual,Path(td))
            self.assertEqual(44,len(manifest['images']))
            self.assertEqual(0,manifest['independent_certified_images'])

    def test_rejects_tampered_hash(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'r.zip';make_zip(p,changed_hash=True)
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                ex.unpack_checked(p,Path(td)/'expanded')

    def test_rejects_missing_file(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'r.zip';make_zip(p,missing=True)
            with self.assertRaisesRegex(ValueError,'identity'):
                ex.unpack_checked(p,Path(td)/'expanded')

    def test_rejects_duplicate_manifest_paths(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'r.zip';make_zip(p,duplicate=True)
            with self.assertRaisesRegex(ValueError,'Repeated image path'):
                ex.unpack_checked(p,Path(td)/'expanded')

    def test_synthetic_summary_explicitly_has_no_accuracy(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'source/images';source.mkdir(parents=True)
            Image.new('RGB',(60,70),'white').save(source/'test.jpg')
            absname=str((source/'test.jpg').resolve())
            manifest=dict(images=[dict(path='images/test.jpg',source='synthetic',source_hint='bottle')])
            decisions={absname:[dict(object_class='bottle',detection_confidence=.9,
                 category='recycling',suggested_category='recycling',reasons=[],
                 box_xyxy_normalized=[.2,.3,.8,.9])]}
            report=ex.summarize(decisions,manifest,root,'test.zip')
            self.assertEqual(1,report['detected_objects_at_display_floor'])
            self.assertEqual({'bottle':1},report['detections_by_object_class'])
            self.assertIsNone(report['map50'])
            self.assertIsNone(report['actual_correct_bin_accuracy'])
            self.assertFalse(report['independent_performance_estimated'])
            self.assertTrue((root/'index.html').is_file())

    def test_frozen_hash_tamper_rejected(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'configs').mkdir();(root/'weights').mkdir()
            protected={}
            paths=['configs/specialists.json','configs/disposal.json',
                   'weights/food-container-carton.pt','weights/plate-wrapper.pt',
                   'weights/utensil.pt','weights/bottle.pt']
            for path in paths:
                p=root/path;p.write_bytes(b'known-fixed-data')
                protected[path]=hashlib.sha256(p.read_bytes()).hexdigest()
            (root/'configs/research-freeze.json').write_text(json.dumps({'protected_sha256':protected}))
            with patch.object(ex.s,'sha',side_effect=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()):
                self.assertEqual(protected,ex.check_frozen(root))
                (root/'weights/bottle.pt').write_bytes(b'altered')
                with self.assertRaisesRegex(ValueError,'hash mismatch'):
                    ex.check_frozen(root)


if __name__=='__main__':unittest.main()
