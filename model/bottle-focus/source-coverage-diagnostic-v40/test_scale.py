from pathlib import Path
import importlib.util
import unittest

spec=importlib.util.spec_from_file_location('v40_scale',Path(__file__).with_name('scale.py'));scale=importlib.util.module_from_spec(spec);spec.loader.exec_module(scale)
class ScaleTests(unittest.TestCase):
 def test_boundary_and_letterbox_geometry(self):
  a=dict(width=1000,height=1000,rows=['0 .5 .5 .05 .05'])
  self.assertEqual(scale.areas_at640(a),[1024.])
  self.assertTrue(scale.eligible(a))
  self.assertFalse(scale.eligible(dict(a,rows=['0 .5 .5 .0499 .05'])))
  self.assertEqual(scale.areas_at640(dict(width=2000,height=1000,rows=['0 .5 .5 .1 .1'])),[2048.])
 def test_all_boxes_in_photo_must_meet_declared_scale(self):
  row=dict(width=1000,height=1000,rows=['0 .5 .5 .2 .2','0 .2 .2 .02 .02'])
  self.assertFalse(scale.eligible(row))
 def test_nonfinite_or_invalid_boxes_rejected(self):
  for rows in [[],['0 .5 .5 nan .1'],['0 .9 .5 .4 .2']]:
   with self.assertRaises((ValueError,AssertionError)):scale.eligible(dict(width=1000,height=1000,rows=rows))
if __name__=='__main__':unittest.main()
