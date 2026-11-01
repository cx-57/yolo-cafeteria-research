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
