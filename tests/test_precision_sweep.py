"""Regression tests: 97% micro-precision goal cannot be claimed by cherry-picking test."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import precision_sweep as p

class PrecisionSweepTests(unittest.TestCase):
    def test_threshold_grid_contains_fixed_baseline_and_high_confidence(self):
        seq=p.thresholds()
        self.assertEqual(.5,seq[0])
        self.assertIn(.97,seq)
        self.assertEqual(.995,seq[-1])
        self.assertEqual(len(seq),len(set(seq)))

    def test_choose_best_recall_with_precision_floor(self):
        rows=[
            dict(threshold=.50,precision=.9342,recall=.7835,predictions=775),
            dict(threshold=.63,precision=.969,recall=.77,predictions=730),
            dict(threshold=.70,precision=.975,recall=.735,predictions=660),
            dict(threshold=.80,precision=.99,recall=.67,predictions=610),
            dict(threshold=.99,precision=1.,recall=.001,predictions=1),
        ]
        chosen=p.select_threshold(rows)
        self.assertEqual(chosen['threshold'],.70)

    def test_no_spurious_97_with_vanishing_coverage(self):
        rows=[
            dict(threshold=.50,precision=.93,recall=.82,predictions=1000),
            dict(threshold=.99,precision=1.0,recall=.02,predictions=10),
        ]
        self.assertIsNone(p.select_threshold(rows))
        self.assertIsNone(p.select_threshold(rows,min_recall=.00,min_predictions=100))

    def test_target_must_be_finite_real(self):
        for target in [0,-1,1.1,float('nan')]:
            if target!=target: # NaN
                # ensure NaN is rejected, not silently interpreted as no threshold
                with self.assertRaises(ValueError):
                    p.select_threshold([],target=target)
            else:
                with self.assertRaises(ValueError):
                    p.select_threshold([],target=target)

    def test_test_results_do_not_select_threshold(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'result.json'
            config={'models':[]}
            freeze={'protected_sha256':{},'manifests':{}}
            manifest=Path(td)/'validation.json'
            test=Path(td)/'test.json'
            manifest.write_text('val');test.write_text('test')
            records=[{'name':'a'}]
            predictions={'a':[]}
            def mock_summary(_records,_preds,threshold):
                if _records[0]['name']=='validation':
                    return dict(threshold=threshold,precision=.93 if threshold<.75 else .98,
                                recall=.82 if threshold<.75 else .64,predictions=200,
                                tp=0,fp=0,fn=0,objects=0,f1=None)
                else:
                    return dict(threshold=threshold,precision=.5 if threshold<.95 else 1.0,
                                recall=.7,predictions=200,tp=0,fp=0,fn=0,objects=0,f1=None)
            with patch.object(p.s,'read',side_effect=[freeze,config]),\
                 patch.object(p.research_audit,'verify_freeze'),\
                 patch.object(p.s,'sha',return_value='fake'),\
                 patch.object(p,'load_cohort',side_effect=[
                     ([{'name':'validation'}],predictions,'valsha'),
                     ([{'name':'test'}],predictions,'testsha')]),\
                 patch.object(p,'summarize',side_effect=mock_summary):
                result=p.run(manifest,test,None,None,out,'cpu',.97,.60,100,False)
            self.assertEqual(.75,result['test']['threshold_selected_without_test'])
            self.assertEqual(.5,result['test']['fixed_threshold_metrics']['precision'])
            self.assertFalse(result['test']['test_precision_at_least_target'])
            self.assertTrue(out.exists())
            self.assertFalse(result['independent_test'])

if __name__=='__main__':unittest.main()
