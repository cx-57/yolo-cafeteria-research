from pathlib import Path
import importlib.util
import unittest
p=Path(__file__).with_name('batches.py');s=importlib.util.spec_from_file_location('batches45test',p);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
class BatchTests(unittest.TestCase):
 def test_real_helper_bounds_input_even_when_predictor_ignores_batch_argument(self):
  class FakeModel:
   def __init__(self):self.sizes=[]
   def predict(self,source,**kwargs):
    # In-memory Ultralytics sources use list length as batch size.
    if len(source)>4:raise AssertionError('GPU batch exceeded4')
    self.sizes.append(len(source));return iter(source)
  model=FakeModel();rows=[dict(image=str(i)) for i in range(142)]
  pairs=list(b.predict_chunks(model,rows))
  self.assertEqual([r['image'] for r,result in pairs],[str(i) for i in range(142)])
  self.assertEqual([result for r,result in pairs],[str(i) for i in range(142)]);self.assertEqual(model.sizes,[4]*35+[2])
if __name__=='__main__':unittest.main()
