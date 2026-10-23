"""Independent known-coordinate checks for non-square rotation frames."""
import unittest
import torch
from pathlib import Path
import hashlib
from ultralytics import YOLO
from rotation_tta import undo_rotation,full_resolution_rotations

class RotationGeometry(unittest.TestCase):
    def test_non_square_box_round_trip_and_score_preservation(self):
        # Original 96x64 frame: interior box and a box touching two edges.
        original=torch.tensor([[[30.,2.],[20.,61.],[10.,4.],[6.,6.],[.8,.3]]])
        views=[
            original.clone(),
            torch.tensor([[[20.,61.],[66.,94.],[6.,6.],[10.,4.],[.8,.3]]]),
            torch.tensor([[[66.,94.],[44.,3.],[10.,4.],[6.,6.],[.8,.3]]]),
            torch.tensor([[[44.,3.],[30.,2.],[6.,6.],[10.,4.],[.8,.3]]]),
        ]
        for k,view in enumerate(views):
            with self.subTest(k=k):
                before=view.clone()
                restored=undo_rotation(view,k,64,96)
                torch.testing.assert_close(restored,original,rtol=0,atol=0)
                torch.testing.assert_close(view,before,rtol=0,atol=0)
                self.assertEqual(restored.dtype,view.dtype)

class NativePredictionIntegrity(unittest.TestCase):
    def test_four_views_keep_identity_scores_and_model_state(self):
        torch.set_num_threads(2)
        weights=Path(__file__).resolve().parent.parent/'head-average-v9/averaged-bottle.pt'
        before=hashlib.sha256(weights.read_bytes()).hexdigest()
        native=YOLO(str(weights)).model.float().cpu().eval()
        state={k:v.clone() for k,v in native.state_dict().items()}
        torch.manual_seed(42);x=torch.rand(1,3,96,128)
        with torch.inference_mode():
            plain=native._predict_once(x)[0].clone()
            combined,_=full_resolution_rotations(native,x)
            count=plain.shape[-1]
            self.assertEqual(combined.shape,(1,5,count*4))
            torch.testing.assert_close(combined[...,:count],plain,rtol=0,atol=0)
            for k in range(4):
                raw=native._predict_once(torch.rot90(x,k,(-2,-1)))[0]
                torch.testing.assert_close(combined[:,4:,k*count:(k+1)*count],raw[:,4:],rtol=0,atol=0)
        for k,v in native.state_dict().items():torch.testing.assert_close(v,state[k],rtol=0,atol=0)
        self.assertEqual(hashlib.sha256(weights.read_bytes()).hexdigest(),before)

if __name__=='__main__':unittest.main()
