import importlib.util
from pathlib import Path
import unittest
p=Path(__file__).with_name('partition.py');s=importlib.util.spec_from_file_location('partition42',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def row():return dict(id='train',image_sha='sha',pixels='pixels',source_group='group')
class PartitionTests(unittest.TestCase):
 def test_monitor_alias_rejected(self):
  for key in ['id','image_sha','pixels','source_group']:
   a=row();b={'id':'monitor','image_sha':'other','pixels':'otherpixel','source_group':'othergroup'};b[key]=a[key]
   with self.assertRaises(AssertionError):m.check([a],[b])
 def test_duplicate_group_rejected(self):
  a=row();b=dict(a,id='different',image_sha='newsha',pixels='newpixel')
  with self.assertRaises(AssertionError):m.check([a,b],[])
 def test_distinct_allowed(self):m.check([row()],[dict(id='m',image_sha='s',pixels='p',source_group='g')])
if __name__=='__main__':unittest.main()
