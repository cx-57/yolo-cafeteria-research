"""Fail closed on changed or unreviewed source data; preserve actual source geometry."""
from pathlib import Path
import copy
import importlib.util
import json
import unittest

RUN=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('round4_admission',RUN/'admission.py')
admission=importlib.util.module_from_spec(spec) if spec else None
if spec and Path(spec.origin).exists():spec.loader.exec_module(admission)

class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(admission,'admit'),'reviewed admission helper missing')
        source=RUN.parent/'external-data-v7'
        self.reviews=json.loads((source/'round4-visual-reviews.json').read_text())
        self.downloads=json.loads((source/'lvis-round4-downloads.json').read_text())
        self.screen=json.loads((source/'round4-duplicate-screen.json').read_text())

    def test_real_accepted_boxes_keep_original_geometry(self):
        result=admission.admit(self.reviews,self.downloads,self.screen)
        self.assertEqual((len(result),sum(len(r['rows']) for r in result)),(17,20))
        r=next(r for r in result if r['id']=='lvis_000000518932')
        _,cx,cy,bw,bh=map(float,r['rows'][0].split())
        self.assertAlmostEqual((cx-bw/2)*427,99.4,places=5)
        self.assertAlmostEqual((cy-bh/2)*640,533.66,places=5)
        self.assertAlmostEqual(bh*640,106.34,places=5)
        self.assertNotIn('lvis_000000476235',{r['id'] for r in result})

    def test_changed_provenance_is_rejected(self):
        for kind in ['screen','image','annotation','duplicate']:
            with self.subTest(kind=kind):
                reviews=copy.deepcopy(self.reviews);downloads=copy.deepcopy(self.downloads);screen=copy.deepcopy(self.screen)
                if kind=='screen':next(r for r in screen['records'] if r['id']==reviews[0]['id'])['hold']=True
                if kind=='image':next(r for r in downloads if r['id']==reviews[0]['id'])['sha']='0'*64
                if kind=='annotation':reviews[0]['source_annotations'][0]['bbox'][0]+=1
                if kind=='duplicate':reviews.append(copy.deepcopy(reviews[0]))
                with self.assertRaises((AssertionError,ValueError)):admission.admit(reviews,downloads,screen)

if __name__=='__main__':unittest.main()
