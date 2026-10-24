from collections import Counter
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from PIL import Image

RUN=Path(__file__).resolve().parent
def module(name):
 spec=importlib.util.spec_from_file_location(name,RUN/f'{name}.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
partition=module('partition');selection=module('selection')

class ControlsTests(unittest.TestCase):
 def test_all_related_views_and_byte_aliases_leave_training(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);paths=[]
   for i,color in enumerate(['blue','red','yellow']):
    p=root/f'{i}.jpg';Image.new('RGB',(24,20),color).save(p);paths.append(str(p))
   alias=root/'alias.jpg';alias.write_bytes(Path(paths[0]).read_bytes())
   with Image.open(paths[0]) as im:pixel=hashlib.sha256(str(im.size).encode()+im.convert('RGB').tobytes()).hexdigest()
   monitor=[dict(source_image=paths[0],image_sha=hashlib.sha256(Path(paths[0]).read_bytes()).hexdigest(),pixels=pixel,source_group='captureA')]
   counts=Counter({paths[0]:4,paths[1]:3,paths[2]:5,str(alias):2});groups={paths[0]:'captureA',paths[1]:'captureA',paths[2]:'captureB',str(alias):'unrecognized_alias'}
   kept,removed=partition.split(counts,groups,monitor)
   self.assertEqual(kept,Counter({paths[2]:5}));self.assertEqual(set(removed),{paths[0],paths[1],str(alias)})
   monitor[0]['image_sha']='changed'
   with self.assertRaises(AssertionError):partition.split(counts,groups,monitor)

 def test_monitor_cannot_count_two_encodings_of_the_same_photo(self):
  with tempfile.TemporaryDirectory() as tmp:
   a=Path(tmp)/'a.jpg';Image.new('RGB',(24,20),'blue').save(a);b=Path(tmp)/'b.jpg';raw=a.read_bytes();b.write_bytes(raw[:2]+b'\xff\xfe\x00\x06test'+raw[2:]);c=Path(tmp)/'c.jpg';Image.new('RGB',(24,20),'red').save(c)
   records=[]
   for p in (a,b):
    digest,pixel=partition.hashes(p);records.append(dict(source_image=str(p),image_sha=digest,pixels=pixel,source_group=p.name))
   with self.assertRaises(AssertionError):partition.split(Counter({str(c):1}),{str(c):'other'},records)

 def test_monitor_scores_cannot_substitute_for_benchmark_and_budget_is_fixed(self):
  baseline=dict(role='initial',selected_epoch=0,AP50=.84,AP50_95=.62,metric_source='continuation_monitor',monitor_fingerprint='monitor')
  better=dict(baseline,role='trained',selected_epoch=4,AP50=.88,AP50_95=.64)
  self.assertEqual(selection.choose([baseline,better],'monitor'),better)
  self.assertEqual(selection.choose([dict(baseline,role='trained',selected_epoch=2),baseline],'monitor'),baseline)
  for wrong in [dict(better,metric_source='fixed_benchmark'),dict(better,monitor_fingerprint='changed'),dict(better,AP50=float('nan'))]:
   with self.assertRaises((AssertionError,ValueError)):selection.choose([baseline,wrong],'monitor')
  self.assertFalse(selection.budget_finished(1));self.assertTrue(selection.budget_finished(9))
  with self.assertRaises(AssertionError):selection.budget_finished(10)

 def test_campaign_declaration_cannot_change_after_monitor_measurement(self):
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'started.json'
   self.assertTrue(selection.bind_campaign(path,'declared'))
   self.assertFalse(selection.bind_campaign(path,'declared'))
   with self.assertRaises(AssertionError):selection.bind_campaign(path,'changed-after-monitor-results')

 def test_final_benchmark_is_bound_to_one_candidate_across_restarts(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'attempt.json'
   self.assertTrue(selection.bind_benchmark(p,'candidateA','dataA'))
   self.assertFalse(selection.bind_benchmark(p,'candidateA','dataA'))
   for weights,data in [('candidateB','dataA'),('candidateA','changed')]:
    with self.assertRaises(AssertionError):selection.bind_benchmark(p,weights,data)

if __name__=='__main__':unittest.main()
