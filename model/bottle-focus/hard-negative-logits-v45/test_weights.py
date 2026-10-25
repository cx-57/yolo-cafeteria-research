from pathlib import Path
import importlib.util
import unittest
p=Path(__file__).with_name('weights.py');s=importlib.util.spec_from_file_location('w45test',p);w=importlib.util.module_from_spec(s);s.loader.exec_module(w)
def row(name,positive=False):return dict(id=name,source_group='group/'+name,image_sha='sha/'+name,pixels='pix/'+name,rows=['0 .5 .5 .2 .2'] if positive else [],exposures=4)
class WeightTests(unittest.TestCase):
 def test_only_confirmed_negative_gets_extra_exposures(self):
  a=row('negative');b=row('positive',True)
  result=w.reweight([a,b],[],[],{'negative'},8);self.assertEqual([r['exposures'] for r in result],[12,4]);self.assertEqual(a['exposures'],4)
  with self.assertRaises(AssertionError):w.reweight([a,b],[],[],{'positive'},8)
 def test_coco_alias_capture_removed_even_with_different_old_group_names(self):
  a=row('lvis_000000001455_crop');held=row('plate__train__lvis_000000001455')
  self.assertEqual(w.reweight([a],[],[held],set(),8),[])
 def test_retention_byte_pixel_and_group_aliases_excluded(self):
  a=row('a');monitor=row('monitor')
  for key in ['image_sha','pixels','source_group']:
   changed=dict(a);changed[key]=monitor[key]
   self.assertEqual(w.reweight([changed],[monitor],[],set(),8),[])
class MiningTests(unittest.TestCase):
 def test_stale_source_or_positive_mining_record_rejected(self):
  parent=row('a');record=dict(parent,predictions=[dict(score=.8)],maximum_false_detection_score=.8)
  w.validate_mining([record],[record],[parent])
  for bad in [dict(record,image_sha='stale'),dict(record,rows=['0 .5 .5 .2 .2'])]:
   with self.assertRaises(AssertionError):w.validate_mining([bad],[bad],[parent])
 def test_candidate_order_must_match_actual_teacher_ranking(self):
  parents=[row('a'),row('b')];records=[dict(p,predictions=[dict(score=score)],maximum_false_detection_score=score) for p,score in zip(parents,[.8,.9])]
  with self.assertRaises(AssertionError):w.validate_mining(records,records,parents)
  w.validate_mining(records,list(reversed(records)),parents)
class LegacyScopeTests(unittest.TestCase):
 def test_only_twice_accepted_original_positive_rows_survive(self):
  a=row('a',True);b=row('b',True)
  first=dict(records=[dict(index=i,id=r['id'],image_sha=r['image_sha'],rows=r['rows'],decision='retain' if i==0 else 'hold') for i,r in enumerate([a,b])],human_verified=False,full_scene_reviewed=True)
  second=dict(accepted_indices=[0],human_verified=False,full_scene_reviewed=True)
  self.assertEqual(w.legacy_holds([a,b],first,second),[b])
  for bad in [dict(second,accepted_indices=[1]),dict(second,accepted_indices=[99])]:
   with self.assertRaises(AssertionError):w.legacy_holds([a,b],first,bad)
  first['records'][0]['rows']=[]
  with self.assertRaises(AssertionError):w.legacy_holds([a,b],first,second)
if __name__=='__main__':unittest.main()
