from collections import Counter
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from PIL import Image

RUN=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('backgrounds',RUN/'backgrounds.py');backgrounds=importlib.util.module_from_spec(spec);spec.loader.exec_module(backgrounds)

class BackgroundTests(unittest.TestCase):
    def fixture(self,tmp):
        root=Path(tmp);images=root/'images/train';labels=root/'labels/train';images.mkdir(parents=True);labels.mkdir(parents=True)
        paths=[];reviews=[]
        for i,color in enumerate(['blue','red','green','yellow']):
            p=images/f'photo{i}.jpg';Image.new('RGB',(20,20),color).save(p);q=labels/f'photo{i}.txt';q.write_text('0 0.5 0.5 0.3 0.3\n' if i==3 else '');paths.append(str(p))
            if i<2:reviews.append(dict(image=str(p),image_sha=hashlib.sha256(p.read_bytes()).hexdigest(),label_sha=hashlib.sha256(q.read_bytes()).hexdigest(),accepted=True,full_scene_reviewed=True,independent_second_review=True,human_verified=False))
        return Counter({paths[0]:2,paths[1]:3,paths[2]:5,paths[3]:7}),reviews,paths

    def test_unknown_empty_images_are_excluded_without_changing_positive_exposures_or_negative_volume(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,reviews,paths=self.fixture(tmp);out=backgrounds.resample(source,reviews,10,seed=43)
            self.assertEqual(out[paths[3]],7);self.assertNotIn(paths[2],out)
            self.assertEqual(out[paths[0]]+out[paths[1]],10)
            self.assertEqual(out,backgrounds.resample(source,list(reversed(reviews)),10,seed=43))
            self.assertEqual(max(out[p] for p in paths[:2])-min(out[p] for p in paths[:2]),0)
            reviews[1]['accepted']=False
            self.assertEqual(backgrounds.resample(source,reviews,10)[paths[0]],10)

    def test_stale_reviews_positive_labels_and_unreviewed_claims_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,reviews,paths=self.fixture(tmp)
            import copy
            for field,value in [('image_sha','wrong'),('label_sha','wrong'),('full_scene_reviewed',False),('independent_second_review',False)]:
                changed=copy.deepcopy(reviews);changed[0][field]=value
                with self.subTest(field=field),self.assertRaises((AssertionError,ValueError)):backgrounds.resample(source,changed,10)
            with self.assertRaises(AssertionError):backgrounds.resample(source,reviews+reviews[:1],10)
            with self.assertRaises(AssertionError):backgrounds.resample(source,reviews,11)
            q=Path(tmp)/'labels/train/photo0.txt';q.write_text('0 0.5 0.5 0.3 0.3\n')
            reviews[0]['label_sha']=hashlib.sha256(q.read_bytes()).hexdigest()
            with self.assertRaises(AssertionError):backgrounds.resample(source,reviews,10)

if __name__=='__main__':unittest.main()
