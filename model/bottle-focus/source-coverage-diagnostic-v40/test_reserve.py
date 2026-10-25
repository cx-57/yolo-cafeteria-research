from pathlib import Path
import importlib.util
import unittest

spec=importlib.util.spec_from_file_location('v40_reserve',Path(__file__).with_name('reserve.py'));reserve=importlib.util.module_from_spec(spec);spec.loader.exec_module(reserve)

def record(i,positive):
 return dict(id=str(i),image_sha='sha'+str(i),pixels='pixels'+str(i),source_group='capture'+str(i),rows=['0 .5 .5 .2 .2'] if positive else [])

class ReservationTests(unittest.TestCase):
 def samples(self):return [record(i,True) for i in range(24)],[record(i,False) for i in range(24,48)]
 def test_fixed_source_sample_does_not_depend_on_input_order(self):
  positive,negative=self.samples();a=reserve.split(positive,negative);b=reserve.split(list(reversed(positive)),list(reversed(negative)))
  self.assertEqual(a,b);self.assertEqual(len(a['monitor']),32);self.assertEqual(len(a['remaining']),16)
  self.assertEqual(sum(bool(r['rows']) for r in a['monitor']),16)
 def test_support_is_not_lowered_when_background_downloads_fail(self):
  positive,negative=self.samples()
  with self.assertRaisesRegex(ValueError,'16 positive and16 negative'):reserve.split(positive,negative[:3])
 def test_old_monitor_id_byte_pixel_or_group_alias_rejected(self):
  positive,negative=self.samples();row=positive[0]
  for forbidden in [dict(forbidden_ids={row['id']}),dict(forbidden_shas={row['image_sha']}),dict(forbidden_pixels={row['pixels']}),dict(forbidden_groups={row['source_group']})]:
   with self.assertRaises(AssertionError):reserve.split(positive,negative,**forbidden)
 def test_related_views_cannot_inflate_monitor_support(self):
  positive,negative=self.samples();positive[1]['source_group']=positive[0]['source_group']
  with self.assertRaises(AssertionError):reserve.split(positive,negative)

if __name__=='__main__':unittest.main()
