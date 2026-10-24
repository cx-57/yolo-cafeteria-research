import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image

RUN=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('oi_admission',RUN/'admission.py');admission=importlib.util.module_from_spec(spec);spec.loader.exec_module(admission)

class AdmissionTests(unittest.TestCase):
    def fixture(self,tmp):
        p=Path(tmp)/'image.jpg';Image.new('RGB',(100,80),'blue').save(p);digest=hashlib.sha256(p.read_bytes()).hexdigest()
        with Image.open(p) as im:pixel=hashlib.sha256(str(im.size).encode()+im.convert('RGB').tobytes()).hexdigest()
        identity='0001abcdef123456';ann=dict(ImageID=identity,Source='xclick',LabelName='/m/04dr76w',Confidence='1',XMin='0.1',XMax='0.3',YMin='0.0625',YMax='0.8125',IsDepiction='0',IsGroupOf='0')
        meta=dict(ImageID=identity,Subset='train',Rotation='0',License='https://creativecommons.org/licenses/by/2.0/',Author='Source author',OriginalLandingURL='https://example.org/photo')
        labels=[dict(ImageID=identity,LabelName='/m/02rlncx',Confidence='1')]
        source=dict(id=identity,official_split='train',image=str(p),sha=digest,pixels=pixel,width=100,height=80,status='unreviewed_generic_bottle',annotations=[ann],photo_metadata=meta,positive_source_labels=labels)
        review=dict(id=identity,accepted=True,image_sha=digest,source_annotations=copy.deepcopy([ann]),annotation_indices=[0],duplicate_screen_sha='screen',photo_metadata_sha=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest(),full_scene_reviewed=True,bottle_scope_reviewed=True,independent_second_review=True,human_verified=False)
        screen=dict(validation_images=246,records=[dict(id=identity,sha=digest,pixels=pixel,hold=False,exact_duplicate=False)])
        return [review],[source],screen,'screen',[dict(id=identity,annotations=copy.deepcopy([ann]),positive_source_labels=copy.deepcopy(labels))],[copy.deepcopy(meta)]

    def test_full_original_boxes_are_normalized_without_relabeling_or_inventing_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(tmp);result=admission.admit(*args)
            self.assertEqual(result[0]['rows'],['0 0.200000000 0.437500000 0.200000000 0.750000000'])
            self.assertEqual(result[0]['annotations'],args[1][0]['annotations'])
            self.assertFalse(result[0]['human_verified'])
            args[0][0]['accepted']=False;self.assertEqual(admission.admit(*args),[])

    def test_missing_boxes_changed_provenance_or_held_sources_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(tmp)
            for case in ['indices','review_sha','pixel','overlap','annotation','metadata','source_group','unreviewed','rotation','groupbox','nonfinite','bounds']:
                x=copy.deepcopy(args)
                if case=='indices':x[0][0]['annotation_indices']=[]
                elif case=='review_sha':x[0][0]['image_sha']='wrong'
                elif case=='pixel':x[2]['records'][0]['pixels']='wrong'
                elif case=='overlap':x[2]['records'][0]['hold']=True
                elif case=='annotation':x[0][0]['source_annotations'][0]['XMax']='0.4'
                elif case=='metadata':x[1][0]['photo_metadata']['License']='unknown'
                elif case=='source_group':x[4][0]['annotations']=[]
                elif case=='unreviewed':x[0][0]['independent_second_review']=False
                elif case=='rotation':
                    x[1][0]['photo_metadata']['Rotation']='90';x[5][0]['Rotation']='90';x[0][0]['photo_metadata_sha']=hashlib.sha256(json.dumps(x[5][0],sort_keys=True).encode()).hexdigest()
                else:
                    key,value={'groupbox':('IsGroupOf','1'),'nonfinite':('XMax','nan'),'bounds':('XMax','1.4')}[case]
                    for row in [x[1][0]['annotations'][0],x[0][0]['source_annotations'][0],x[4][0]['annotations'][0]]:row[key]=value
                with self.subTest(case=case),self.assertRaises((AssertionError,ValueError)):admission.admit(*x)
            Path(args[1][0]['image']).write_bytes(b'changed')
            with self.assertRaises((AssertionError,ValueError)):admission.admit(*args)

    def test_repeated_reviews_and_prior_photo_hashes_or_pixels_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.fixture(tmp)
            with self.assertRaises(AssertionError):admission.admit(args[0]*2,*args[1:])
            with self.assertRaises(AssertionError):admission.admit([], *args[1:])
            with self.assertRaises(AssertionError):admission.admit(*args,forbidden_shas={args[1][0]['sha']})
            with self.assertRaises(AssertionError):admission.admit(*args,forbidden_pixels={args[1][0]['pixels']})

if __name__=='__main__':unittest.main()
