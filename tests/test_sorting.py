"""Disposal policy, prediction replay, and firmware protocol tests."""

# --- test_disposal.py ---
import copy
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import disposal as d

class DisposalTests(unittest.TestCase):
    def setUp(self):
        self.policy=json.loads(d.DEFAULT_RULES.read_text())

    def test_unverified_policy_never_returns_confirmed_bin(self):
        self.policy['verified']=False
        for name in d.s.NAMES:
            result=d.classify(name,.99,self.policy,{'empty':True})
            self.assertEqual(result['category'],'review')
            self.assertTrue(result['needs_review'])

    def test_food_and_bottle_sort_without_empty_confirmation(self):
        self.policy['verified']=True
        self.assertEqual(d.classify('food',.9,self.policy)['category'],'compost')
        for context in [None, {'empty':True}, {'empty':False}, {'empty':'true'}]:
            self.assertEqual(d.classify('bottle',.9,self.policy,context)['category'],'recycling')

    def test_unknown_material_reusable_and_low_confidence_require_review(self):
        self.policy['verified']=True
        for name in ['plate','utensil','hinged_container','wrapper','unknown']:
            self.assertEqual(d.classify(name,.99,self.policy)['category'],'review')
        result=d.classify('food',.3,self.policy)
        self.assertEqual(result['category'],'review')
        self.assertEqual(result['detection_confidence'],.3)
        self.assertNotIn('disposal_confidence',result)

    def test_mixed_scene_preserves_individual_boxes_and_empty_scene(self):
        self.policy['verified']=True
        rows=[[0,0,.5,.5,.9,0],[.5,.5,1,1,.8,6],[0,0,1,1,.01,1]]
        original=copy.deepcopy(rows)
        result=d.classify_predictions({'meal':rows,'empty':[]},self.policy,contexts={'meal':{'1':{'empty':True}}})
        self.assertEqual([r['category'] for r in result['meal']],['compost','recycling'])
        self.assertEqual(result['empty'],[])
        self.assertEqual(result['meal'][1]['box_xyxy_normalized'],[.5,.5,1,1])
        self.assertEqual(rows,original)

    def test_bottle_and_carton_are_independent_of_empty_context(self):
        rows=[[0,0,.5,.5,.9,6],[.5,.5,1,1,.9,4]]
        result=d.classify_predictions({'x':rows},self.policy,contexts={'x':{'0':{'empty':True},'1':{'empty':False}}})
        self.assertEqual([r['category'] for r in result['x']],['recycling','recycling'])
        no_context=d.classify_predictions({'x':rows},self.policy)
        self.assertEqual([r['category'] for r in no_context['x']],['recycling','recycling'])

    def test_cartons_recycle_without_empty_confirmation(self):
        for context in [None, {'empty':True}, {'empty':False}]:
            self.assertEqual(d.classify('drink_carton',.9,self.policy,context)['category'],'recycling')
        self.assertEqual(d.classify('drink_carton',.49,self.policy)['category'],'review')
        self.assertEqual(d.classify('bottle',.49,self.policy)['category'],'review')

    def test_invalid_policy_and_prediction_fail_explicitly(self):
        bad=copy.deepcopy(self.policy);bad['classes']['bottle']['category']='made_up'
        with self.assertRaises(ValueError):d.validate_policy(bad)
        bad=copy.deepcopy(self.policy);bad['verified']='yes'
        with self.assertRaises(ValueError):d.validate_policy(bad)
        for value in [float('nan'),-.1,1.1]:
            with self.assertRaises(ValueError):d.classify('food',value,self.policy)
        with self.assertRaises(ValueError):d.classify_predictions({'x':[[0,0,1,1,.9,99]]},self.policy)

if __name__=='__main__':unittest.main()


# --- test_replay_sorting.py ---
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import replay_sorting as r
import specialists as s


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


# --- test_sorter_firmware.py ---
"""Compile the actual .ino against a host clock/Serial/Servo harness; no hardware."""
import shutil
import subprocess


class FirmwareTests(unittest.TestCase):
    def test_protocol_and_timed_sequence_on_all_three_positions(self):
        compiler = shutil.which('clang++') or shutil.which('g++')
        if not compiler:
            self.skipTest('Host C++ compiler unavailable; install clang or g++')
        source = Path(__file__).resolve().parent/'firmware'
        with tempfile.TemporaryDirectory() as td:
            exe = str(Path(td)/'sequence')
            flags = []
            if sys.platform == 'darwin':
                sdk = subprocess.check_output(['xcrun', '--show-sdk-path'], text=True).strip()
                flags = ['-isystem', str(Path(sdk)/'usr/include/c++/v1')]
            subprocess.run([compiler, *flags, '-std=c++17', '-Wall', '-Wextra', '-Werror', '-I', str(source),
                            str(source/'sequence.cpp'), '-o', exe], check=True, capture_output=True, text=True)
            subprocess.run([exe], check=True, capture_output=True, text=True, timeout=10)
