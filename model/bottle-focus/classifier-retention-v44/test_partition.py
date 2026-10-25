from pathlib import Path
import importlib.util
import unittest
p=Path(__file__).with_name('partition.py');spec=importlib.util.spec_from_file_location('p44test',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def row(i):return dict(id=str(i),image_sha='sha'+str(i),pixels='pix'+str(i),source_group='grp'+str(i),rows=['0 .5 .5 .2 .2'])
class PartitionTests(unittest.TestCase):
 def test_every_reserved_identity_alias_is_removed(self):
  monitor=row(0)
  for key in ['id','image_sha','pixels','source_group']:
   candidate=row(1);candidate[key]=monitor[key]
   self.assertEqual(m.keep([candidate],[monitor],set(),set()),[])
 def test_held_groups_and_screen_flags_removed(self):
  self.assertEqual(m.keep([row(1),row(2)],[],{'grp1'},{'2'}),[])
 def test_feather_correction_needs_both_reviews_and_exact_source(self):
  original=row(39);first=dict(index=39,image_sha=original['image_sha'],rows=original['rows'],approved_rows=[],decision='correct_to_negative',verified_absence=True)
  second=dict(accepted_indices=[39],negative_indices=[39],training_only_correction_indices=[39],full_scene_reviewed=True,human_verified=False)
  self.assertEqual(m.reviewed_rows(original,first,second),[])
  for bad in [dict(second,negative_indices=[]),dict(second,training_only_correction_indices=[]),dict(second,accepted_indices=[])]:
   with self.assertRaises(AssertionError):m.reviewed_rows(original,first,bad)
  with self.assertRaises(AssertionError):m.reviewed_rows(dict(original,image_sha='changed'),first,second)
 def test_positive_box_edits_are_rejected(self):
  original=row(1);first=dict(index=1,image_sha=original['image_sha'],rows=original['rows'],approved_rows=['0 .4 .4 .1 .1'],decision='retain')
  with self.assertRaises(AssertionError):m.reviewed_rows(original,first,dict(accepted_indices=[1],full_scene_reviewed=True,human_verified=False))
if __name__=='__main__':unittest.main()
