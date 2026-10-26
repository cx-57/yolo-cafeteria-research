import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import campaign

class GateLifecycleTests(unittest.TestCase):
 def execute(self,values):
  base=[.9,.8,.6,.5,.4,.3]
  baseline=dict(role='initial',epoch=0,weights_sha=campaign.parent_run.INITIAL_SHA,values=base)
  saved={};evaluations=[]
  class Locks:
   def bind_campaign(self,*a):pass
   def bind_benchmark(self,*a):saved['benchmark_bound']=a
  def evaluate(path,cohort):
   evaluations.append(cohort['name'])
   if cohort['name']=='original-benchmark':return dict(AP50=.88,AP50_95=.64)
   idx={'retention':0,'openimages':2,'litter':4}[cohort['name']]
   return dict(AP50=values[idx],AP50_95=values[idx+1])
  with tempfile.TemporaryDirectory() as tmp,patch.object(campaign,'RUN',Path(tmp)),patch.object(campaign,'verify',return_value={}),patch.object(campaign,'sha',return_value='sha'),patch.object(campaign,'load',return_value=dict(reference=base,chosen=baseline)),patch.object(campaign,'save',side_effect=lambda p,r:saved.update({p.name:r})),patch.object(campaign.data.parent.sources,'module',return_value=Locks()),patch.object(campaign,'build',return_value=Path(tmp)/'model.pt'),patch.object(campaign,'cohorts',return_value=[dict(name=n) for n in ['retention','openimages','litter']]),patch.object(campaign,'evaluate',side_effect=evaluate),patch.object(campaign,'frozen_benchmark',return_value={}):
   campaign.main()
  return saved,evaluations
 def test_failed_gate_never_binds_or_evaluates_benchmark(self):
  saved,calls=self.execute([.9,.8,.8,.6,.5,.299])
  self.assertNotIn('benchmark_bound',saved);self.assertNotIn('original-benchmark',calls)
  self.assertFalse(saved['campaign-result.json']['original_benchmark_used'])
 def test_two_eligible_candidates_use_one_benchmark_and_preserve_tie_order(self):
  saved,calls=self.execute([.9,.8,.8,.6,.5,.3])
  self.assertEqual(calls.count('original-benchmark'),1)
  self.assertEqual(saved['campaign-result.json']['chosen']['alpha'],.25)
  self.assertTrue(saved['campaign-result.json']['promotion_eligible'])
  self.assertFalse(saved['campaign-result.json']['move_on_eligible'])
if __name__=='__main__':unittest.main()
