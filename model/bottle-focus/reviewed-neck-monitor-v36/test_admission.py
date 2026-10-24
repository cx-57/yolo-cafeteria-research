import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from PIL import Image

RUN=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v36_admission',RUN/'admission.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

class AdmissionTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.image=Path(self.tmp.name)/'photo.jpg';Image.new('RGB',(100,80),'blue').save(self.image)
  self.sha,self.pixel=a.hashes(self.image)
  self.annotation=dict(ImageID='new',LabelName='/m/04dr76w',Confidence='1',IsGroupOf='0',IsDepiction='0',XMin='.1',XMax='.8',YMin='.2',YMax='.9')
  self.metadata=dict(ImageID='new',Subset='train',Rotation='0.0',License='https://creativecommons.org/licenses/by/2.0/',Author='author',OriginalLandingURL='https://example.test/photo')
  self.source=dict(id='new',source_type='openimages',role='positive',image=str(self.image),sha=self.sha,width=100,height=80,annotations=[self.annotation],photo_metadata=self.metadata)
  self.official=dict(annotations=[copy.deepcopy(self.annotation)],photo_metadata=copy.deepcopy(self.metadata))
  self.review=dict(accepted=True,full_scene_reviewed=True,bottle_scope_reviewed=True,independent_second_review=True,human_verified=False,annotation_indices=[0],source_annotations=copy.deepcopy(self.source['annotations']),image_sha=self.sha,metadata_sha=a.json_sha(self.metadata))
  self.screen=dict(id='new',sha=self.sha,pixels=self.pixel,hold=False)
 def admit(self,**kwargs):
  return a.admit(self.source,self.review,self.official,self.screen,**kwargs)
 def test_valid_whole_photo_admits_original_box(self):
  self.assertEqual(self.admit()['rows'],['0 0.450000000 0.550000000 0.700000000 0.700000000'])
 def test_source_box_tampering_and_partial_annotations_rejected(self):
  self.source['annotations'][0]['XMax']='.7'
  with self.assertRaises(AssertionError):self.admit()
  self.source['annotations']=copy.deepcopy(self.official['annotations']);self.review['annotation_indices']=[]
  with self.assertRaises(AssertionError):self.admit()
 def test_permanent_monitor_id_byte_and_pixel_aliases_rejected(self):
  for kwargs in [dict(forbidden_ids={'new'}),dict(forbidden_shas={self.sha}),dict(forbidden_pixels={self.pixel})]:
   with self.assertRaises(AssertionError):self.admit(**kwargs)
 def test_screen_hold_or_changed_identity_rejected(self):
  self.screen['hold']=True
  with self.assertRaises(AssertionError):self.admit()
  self.screen['hold']=False;self.screen['sha']='wrong'
  with self.assertRaises(AssertionError):self.admit()
 def test_negative_requires_explicit_exhaustive_absence_review(self):
  self.source['role']='negative'
  with self.assertRaises(AssertionError):self.admit()
  self.review.update(annotation_indices=[],verified_absence=True,all_source_boxes_non_target=True)
  self.assertEqual(self.admit()['rows'],[])
 def test_taco_requires_complete_original_clear_bottle_annotations(self):
  box=dict(id=1,image_id=1,category_id=5,bbox=[10,16,70,56])
  metadata=dict(photo_id='123',license_url='https://creativecommons.org/licenses/by/2.0/',author={'name':'author'},photo_page='https://www.flickr.com/photos/author/123')
  self.source.update(source_type='taco',annotations=[box],all_source_annotations=[box],photo_metadata=metadata,original=dict(id=1,width=100,height=80,flickr_url='https://farm.test/123_hash_o.jpg'))
  self.official=dict(annotations=copy.deepcopy([box]),all_source_annotations=copy.deepcopy([box]),photo_metadata=copy.deepcopy(metadata),original=copy.deepcopy(self.source['original']))
  self.review.update(source_annotations=copy.deepcopy([box]),metadata_sha=a.json_sha(metadata))
  self.assertEqual(self.admit()['rows'],['0 0.450000000 0.550000000 0.700000000 0.700000000'])
  self.official['all_source_annotations'].append(dict(box,id=2,category_id=4))
  self.source['all_source_annotations']=copy.deepcopy(self.official['all_source_annotations'])
  with self.assertRaises(AssertionError):self.admit()

if __name__=='__main__':unittest.main()
