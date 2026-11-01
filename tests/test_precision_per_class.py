"""Classwise confidence optimizer must not hide recall failures or modify models."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import precision_per_class as q

class PrecisionPerClassTests(unittest.TestCase):
    def make_rows(self):
        result=[]
        for t in (.50,.70):
            per={}
            for i,name in enumerate(q.CLASS_NAMES):
                if i==0:
                    tp,fp=(30,15) if t==.5 else (25,2)
                else:
                    tp,fp=(30,0) if t==.5 else (20,0)
                per[name]=dict(tp=tp,fp=fp,fn=40-tp,objects=40,predictions=tp+fp,
                               precision=tp/(tp+fp),recall=tp/40)
            result.append(dict(threshold=t,per_class=per))
        return result

    def test_classwise_combination_can_improve_precision_without_wholesale_recall_loss(self):
        chosen=q.optimize(self.make_rows(),goal=.97,min_recall=.60,min_predictions=100)['selected']
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen['tp'],205)
        self.assertEqual(chosen['fp'],2)
        self.assertGreater(chosen['precision'],.97)
        self.assertGreater(chosen['recall'],.70)
        self.assertEqual(chosen['thresholds']['food'],.70)
        self.assertEqual(chosen['thresholds']['plate'],.50)

    def test_recalls_and_prediction_guards_cannot_be_bypassed(self):
        self.assertIsNone(q.optimize(self.make_rows(),min_recall=.95)['selected'])
        self.assertIsNone(q.optimize(self.make_rows(),min_predictions=300)['selected'])

    def test_no_97_at_any_threshold_returns_no_qualified_solution(self):
        rows=self.make_rows()
        for r in rows:
            for name,metrics in r['per_class'].items():
                metrics['fp']+=20
        self.assertIsNone(q.optimize(rows,goal=.97)['selected'])

    def test_per_class_filter_applies_separate_cutoffs(self):
        names=q.CLASS_NAMES
        rows=['0 .2 .2 .2 .2','1 .7 .7 .2 .2']
        records=[dict(name='a',known_classes=list(range(7)),rows=rows)]
        pred={'a':[[.1,.1,.3,.3,.7,0],[.6,.6,.8,.8,.6,1]]}
        thresholds=dict.fromkeys(names,.5)
        baseline=q.evaluate_thresholds(records,pred,thresholds)
        self.assertEqual((baseline['tp'],baseline['fp']), (2,0))
        thresholds['food']=.8
        changed=q.evaluate_thresholds(records,pred,thresholds)
        self.assertEqual((changed['tp'],changed['fn']), (1,1))

    def test_invalid_threshold_vectors_rejected(self):
        with self.assertRaises(ValueError):
            q.evaluate_thresholds([],{},dict(food=.6))
        with self.assertRaises(ValueError):
            q.optimize(self.make_rows(),goal=float('nan'))

if __name__=='__main__':
    unittest.main()
