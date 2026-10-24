import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image

RUN=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v31_admission',RUN/'admission.py')
admission=importlib.util.module_from_spec(spec);spec.loader.exec_module(admission)

class AdmissionTests(unittest.TestCase):
    def fixture(self,tmp):
        p=Path(tmp)/'source.jpg';Image.new('RGB',(100,80),'blue').save(p)
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        with Image.open(p) as im:pixel=hashlib.sha256(str(im.size).encode()+im.convert('RGB').tobytes()).hexdigest()
        ann=dict(id=1,image_id=42,category_id=1162,bbox=[10.,5.,20.,60.])
        source=dict(id='lvis_000000000042',image=str(p),sha=digest,status='unreviewed',original=dict(id=42,width=100,height=80,not_exhaustive_category_ids=[]),annotations=[ann])
        review=dict(id=source['id'],accepted=True,image_sha=digest,source_annotations=[copy.deepcopy(ann)],annotation_indices=[0],duplicate_screen_sha='screen-hash',human_verified=False)
        screened=dict(id=source['id'],sha=digest,pixels=pixel,hold=False,exact_duplicate=False)
        return [review],[source],dict(validation_images=246,records=[screened])

    def test_real_image_and_all_source_boxes_admitted_without_relabeling(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(tmp);result=admission.admit(*args,'screen-hash')
            self.assertEqual(len(result),1)
            self.assertEqual(result[0]['rows'],['0 0.200000000 0.437500000 0.200000000 0.750000000'])
            self.assertFalse(result[0]['human_verified'])
            args[0][0]['accepted']=False
            self.assertEqual(admission.admit(*args,'screen-hash'),[])

    def test_changed_bytes_annotations_or_screen_binding_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(tmp)
            for change in ['bytes','annotations','screen','indices','overlap','pixel','bounds','nonfinite','nonexhaustive','dimensions']:
                value=copy.deepcopy(args)
                if change=='bytes':value[0][0]['image_sha']='wrong'
                elif change=='annotations':value[0][0]['source_annotations'][0]['bbox'][0]=11
                elif change=='screen':value[0][0]['duplicate_screen_sha']='wrong'
                elif change=='indices':value[0][0]['annotation_indices']=[]
                elif change=='overlap':value[2]['records'][0]['hold']=True
                elif change=='pixel':value[2]['records'][0]['pixels']='wrong'
                elif change in ('bounds','nonfinite'):
                    value[1][0]['annotations'][0]['bbox'][2]=200 if change=='bounds' else float('nan')
                    value[0][0]['source_annotations']=copy.deepcopy(value[1][0]['annotations'])
                elif change=='nonexhaustive':value[1][0]['original']['not_exhaustive_category_ids']=[1162]
                elif change=='dimensions':value[1][0]['original']['width']=101
                with self.subTest(change=change),self.assertRaises((AssertionError,ValueError)):
                    admission.admit(*value,'screen-hash')
            Path(args[1][0]['image']).write_bytes(b'changed')
            with self.assertRaises((AssertionError,ValueError)):admission.admit(*args,'screen-hash')

    def test_duplicates_missing_reviews_and_protected_sources_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(tmp);review,source,screen=args
            for bad in [(review*2,source,screen),([],source,screen),(review,source*2,screen)]:
                with self.assertRaises(AssertionError):admission.admit(*bad,'screen-hash')
            with self.assertRaises(AssertionError):admission.admit(*args,'screen-hash',forbidden_shas={source[0]['sha']})
            with self.assertRaises(AssertionError):admission.admit(*args,'screen-hash',forbidden_pixels={screen['records'][0]['pixels']})
            second=copy.deepcopy(source[0]);second['id']='second';other=copy.deepcopy(review[0]);other['id']='second';checked=copy.deepcopy(screen['records'][0]);checked['id']='second'
            with self.assertRaises(AssertionError):admission.admit(review+[other],source+[second],dict(validation_images=246,records=screen['records']+[checked]),'screen-hash')

if __name__=='__main__':unittest.main()
