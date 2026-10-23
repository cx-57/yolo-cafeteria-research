"""Integrity and learned-head transfer for the two-class specialist."""
import importlib.util
import copy
from pathlib import Path
import sys
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

try:
    import utensil_packet as specialist
except ImportError:
    specialist = None


ML_AVAILABLE = bool(importlib.util.find_spec('torch') and importlib.util.find_spec('ultralytics'))


class SpecialistTests(unittest.TestCase):
    def test_local_file_read_accepts_stat_without_macos_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'source.bin'
            path.write_bytes(b'payload')
            with patch.object(Path, 'stat', return_value=SimpleNamespace(st_size=7)):
                self.assertEqual(specialist.read_bytes(path), b'payload')

    def test_offloaded_file_is_rejected_before_read(self):
        with patch.object(Path, 'stat', return_value=SimpleNamespace(st_flags=0x40000000)), \
                patch.object(Path, 'read_bytes', side_effect=AssertionError('must not read')):
            with self.assertRaisesRegex(ValueError, 'iCloud'):
                specialist.read_bytes(Path('/offloaded-source'))

    def test_small_offloaded_json_is_materialized_with_complete_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'selection.json';payload=b'{"epoch": 3}\n';path.write_bytes(payload)
            stat=SimpleNamespace(st_flags=0x40000000,st_size=len(payload))
            with patch.object(Path,'stat',return_value=stat):
                self.assertEqual(specialist.load(path),{'epoch':3})
                self.assertEqual(specialist.read_bytes(path),payload)
            with patch.object(Path,'stat',return_value=SimpleNamespace(st_flags=0x40000000,st_size=len(payload)+1)):
                with self.assertRaisesRegex(ValueError,'Incomplete'):specialist.read_bytes(path)

    def test_large_offloaded_json_still_rejects_automatic_download(self):
        stat=SimpleNamespace(st_flags=0x40000000,st_size=2*1024*1024)
        with patch.object(Path,'stat',return_value=stat),patch.object(Path,'read_bytes',side_effect=AssertionError('must not download large metadata')):
            with self.assertRaisesRegex(ValueError,'iCloud'):specialist.read_bytes(Path('/large-audit.json'))

    def test_packet_selection_rejects_regressing_below_utensil_milestone(self):
        self.assertEqual(specialist.packet_selection_score([.49,.8]),-1.)
        self.assertAlmostEqual(specialist.packet_selection_score([.71,.4]),.4)
        self.assertEqual(specialist.packet_selection_score([.8]),-1.)
        self.assertEqual(specialist.packet_selection_score([.8,float('nan')]),-1.)

    def test_packet_repetition_only_repeats_training_positives_with_bound_ancestry(self):
        rows = [dict(name='packet.jpg',split='train',sha='packet-sha',group='capture-a',rows=['1 .5 .5 .2 .2']),
                dict(name='negative.jpg',split='train',sha='negative-sha',rows=[]),
                dict(name='held.jpg',split='val',sha='held-sha',rows=['1 .5 .5 .2 .2'])]
        before = copy.deepcopy(rows)
        result = specialist.repeat_packet_records(rows,Path('/source'),copies=4)
        self.assertEqual(rows,before)
        self.assertEqual(len(result),6)
        self.assertEqual([r for r in result if r['split']=='val'],[rows[-1]])
        for r in result[3:]:
            self.assertEqual(r['sha'],'packet-sha')
            self.assertEqual(r['group'],'capture-a')
            self.assertEqual(r['rows'],rows[0]['rows'])
            self.assertEqual(r['parents'],[dict(name='packet.jpg',sha='packet-sha',split='train')])
            self.assertEqual(r['image_path'],'/source/images/train/packet.jpg')
        self.assertEqual(len({r['name'] for r in result}),6)

    @unittest.skipUnless(ML_AVAILABLE, 'requires ML dependencies; covered by ml-integration')
    def test_coco_restart_keeps_backbone_and_uses_two_class_head(self):
        import torch
        from ultralytics.nn.tasks import DetectionModel
        torch.set_num_threads(1)
        source = DetectionModel('yolo11n.yaml', nc=80, verbose=False).float()
        with torch.no_grad():
            source.model[0].conv.weight.fill_(0.123)
        result = specialist.coco_restart_model(source)
        self.assertEqual(result.model[-1].nc, 2)
        self.assertEqual(result.names, dict(enumerate(specialist.NAMES)))
        torch.testing.assert_close(result.model[0].conv.weight, source.model[0].conv.weight)
        self.assertEqual(source.model[-1].nc, 80)

    def test_training_correction_rejects_validation_and_hash_mismatch(self):
        self.assertIsNotNone(specialist)
        original=[dict(name='spoon.jpg',split='train',sha='bound',rows=[],verified_absence=True),dict(name='held.jpg',split='val',sha='held',rows=[],verified_absence=False)]
        review=dict(name='spoon.jpg',sha='bound',expected_rows=[],rows=['0 .5 .5 .4 .1'],reviewer='visual audit',reason='Missed spoon')
        corrected=specialist.apply_training_corrections(original,[review])
        self.assertEqual(original[0]['rows'],[])
        self.assertEqual(corrected[0]['rows'],review['rows'])
        self.assertFalse(corrected[0]['verified_absence'])
        self.assertEqual(corrected[1],original[1])
        with self.assertRaises(ValueError):specialist.apply_training_corrections(original,[dict(review,sha='changed')])
        with self.assertRaises(ValueError):specialist.apply_training_corrections(original,[dict(review,name='held.jpg',sha='held')])

    def test_composite_boxes_match_rendered_object_and_reject_unsafe_parents(self):
        self.assertIsNotNone(specialist)
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);background=root/'background.png';packet=root/'packet.png'
            Image.new('RGB',(200,160),'black').save(background)
            Image.new('RGB',(80,40),'white').save(packet)
            bg=dict(name='bg.png',split='train',rows=[],known_classes=[0,1],verified_absence=True,image_path=str(background),sha=specialist.sha(background))
            obj=dict(name='packet.png',split='train',rows=['1 .5 .5 1 1'],known_classes=[0,1],image_path=str(packet),sha=specialist.sha(packet),polygon=[(0,0),(79,0),(79,39),(0,39)])
            result=specialist.render_composite(bg,obj,root/'out.png',seed=42)
            im=Image.open(root/'out.png').convert('RGB');box=im.getbbox()
            c,x,y,w,h=map(float,result['rows'][0].split())
            self.assertEqual(c,1)
            expected=((box[0]+box[2])/400,(box[1]+box[3])/320,(box[2]-box[0])/200,(box[3]-box[1])/160)
            for a,b in zip((x,y,w,h),expected):self.assertAlmostEqual(a,b,places=7)
            self.assertEqual(result['split'],'train')
            self.assertEqual({p['name'] for p in result['parents']},{'bg.png','packet.png'})
            with self.assertRaises(ValueError):specialist.render_composite(dict(bg,split='val'),obj,root/'bad.png',seed=42)
            with self.assertRaises(ValueError):specialist.render_composite(dict(bg,rows=['1 .5 .5 .2 .2']),obj,root/'bad.png',seed=42)

    def test_balance_caps_repeated_capture_without_removing_validation(self):
        self.assertIsNotNone(specialist)
        names=['464122','464135','464147','464164']+[str(i) for i in range(40)]
        records=[dict(name=f'condiments__train__{n}_0.jpg',split='train',rows=['1 .5 .5 .2 .2']) for n in names]
        records += [dict(name='val.jpg',split='val',rows=['1 .5 .5 .2 .2'])]
        original=copy.deepcopy(records)
        balanced=specialist.balance_packets(records)
        self.assertEqual(len(balanced),5)
        self.assertEqual(balanced[-1],records[-1])
        self.assertEqual(records,original)

    def test_dataset_configuration_and_membership_are_bound(self):
        self.assertIsNotNone(specialist)
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'data.yaml').write_text(yaml.safe_dump(dict(path=str(root),train='images/train',val='images/val',names=specialist.NAMES)))
            for split in ('train','val'):
                (root/'images'/split).mkdir(parents=True)
                (root/'labels'/split).mkdir(parents=True)
            specialist.validate_configuration(root, [])
            (root/'images/train/unreviewed.jpg').write_bytes(b'new unreviewed photo')
            with self.assertRaises(ValueError): specialist.validate_configuration(root, [])
            (root/'images/train/unreviewed.jpg').unlink()
            (root/'images/train/nested').mkdir()
            with self.assertRaises(ValueError): specialist.validate_configuration(root, [])
            (root/'images/train/nested').rmdir()
            (root/'data.yaml').write_text(yaml.safe_dump(dict(path='/different/dataset',train='images/train',val='images/val',names=specialist.NAMES)))
            with self.assertRaises(ValueError): specialist.validate_configuration(root, [])

    def test_projection_requires_both_training_reviews_and_keeps_all_validation(self):
        self.assertIsNotNone(specialist, 'Need a reviewed two-class specialist')
        records = [dict(name='both.jpg', split='train', rows=['3 .5 .5 .2 .3','7 .2 .2 .1 .1','0 .5 .5 .9 .9']),
                   dict(name='partial.jpg', split='train', rows=['3 .5 .5 .2 .3']),
                   dict(name='negative.jpg', split='train', rows=[]),
                   dict(name='val-positive.jpg', split='val', rows=['7 .2 .2 .1 .1']),
                   dict(name='val-unknown.jpg', split='val', rows=[])]
        original=copy.deepcopy(records)
        projected=specialist.project_records(records, {'both.jpg':(3,7),'partial.jpg':(3,),
            'negative.jpg':(3,7)}, set())
        self.assertEqual(records,original)
        self.assertEqual([r['name'] for r in projected],['both.jpg','negative.jpg','val-positive.jpg','val-unknown.jpg'])
        self.assertEqual(projected[0]['rows'],['0 .5 .5 .2 .3','1 .2 .2 .1 .1'])
        self.assertTrue(projected[1]['verified_absence'])
        self.assertFalse(projected[-1]['verified_absence'])
        self.assertEqual(specialist.project_records(records,{'both.jpg':(3,7)}, {'both.jpg'}),
                         [r for r in projected if r['split']=='val'])

    @unittest.skipUnless(ML_AVAILABLE, 'requires ML dependencies; covered by ml-integration')
    def test_projection_preserves_decoded_boxes_and_target_scores(self):
        self.assertIsNotNone(specialist, 'Need learned output-channel projection')
        import torch
        from ultralytics.nn.tasks import DetectionModel
        from train import class_names
        torch.set_num_threads(1)
        model=DetectionModel('yolo11n.yaml',nc=11,verbose=False).float().eval()
        model.names=dict(enumerate(class_names()))
        projected=specialist.project_model(model).eval()
        self.assertEqual(model.model[-1].nc,11)
        self.assertEqual(projected.model[-1].nc,2)
        self.assertEqual(projected.names,{0:'utensil',1:'condiment_packet'})
        x=torch.rand(1,3,64,64)
        with torch.no_grad():
            old=model(x)[0];new=projected(x)[0]
        torch.testing.assert_close(new[:,:4],old[:,:4],rtol=0,atol=0)
        torch.testing.assert_close(new[:,4:],old[:,[7,11]],rtol=1e-5,atol=1e-7)

    def test_training_coverage_cannot_cross_into_validation(self):
        self.assertIsNotNone(specialist)
        source=[dict(name='same.jpg',split='val',rows=['3 .4 .5 .2 .3'])]
        projected=specialist.project_records(source,{'same.jpg':(3,7)},set())
        self.assertEqual(projected[0]['split'],'val')
        self.assertFalse(projected[0]['verified_absence'])
        self.assertEqual(projected[0]['rows'],['0 .4 .5 .2 .3'])

if __name__=='__main__':
    unittest.main()
