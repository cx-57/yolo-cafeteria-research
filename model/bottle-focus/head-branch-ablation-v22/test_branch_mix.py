"""Check actual decoded outputs against intended independent head sources."""
from pathlib import Path
import hashlib,unittest
import torch
from ultralytics import YOLO
from branch_mix import mix_branches

class HeadBranchBehavior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        base=Path(__file__).resolve().parent.parent
        cls.paths=[base/'head-interpolation-v16/alpha0.75-bottle.pt',base/'quality-head-blend-v19/alpha0.25-bottle.pt']
        cls.hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in cls.paths]
        cls.a,cls.b=[YOLO(str(p)).model.float().cpu().eval() for p in cls.paths]
        torch.manual_seed(42);cls.x=torch.rand(1,3,96,128)
        with torch.inference_mode():
            cls.pred_a=cls.a._predict_once(cls.x)[0].clone();cls.pred_b=cls.b._predict_once(cls.x)[0].clone()

    def test_regression_only_preserves_initial_scores_and_quality_boxes(self):
        mixed=mix_branches(self.a,self.b,'regression')
        with torch.inference_mode():out=mixed._predict_once(self.x)[0]
        torch.testing.assert_close(out[:,:4],self.pred_b[:,:4],rtol=0,atol=0)
        torch.testing.assert_close(out[:,4:],self.pred_a[:,4:],rtol=0,atol=0)
        self.assertEqual([hashlib.sha256(p.read_bytes()).hexdigest() for p in self.paths],self.hashes)

    def test_classification_only_preserves_initial_boxes_and_quality_scores(self):
        mixed=mix_branches(self.a,self.b,'classification')
        with torch.inference_mode():out=mixed._predict_once(self.x)[0]
        torch.testing.assert_close(out[:,:4],self.pred_a[:,:4],rtol=0,atol=0)
        torch.testing.assert_close(out[:,4:],self.pred_b[:,4:],rtol=0,atol=0)
        self.assertEqual([hashlib.sha256(p.read_bytes()).hexdigest() for p in self.paths],self.hashes)

if __name__=='__main__':unittest.main()
