"""Known fusion examples and integration with native detection postprocessing."""
import unittest
import torch
from fusion import fuse_views,FlipFusionValidator

def view(boxes=(),scores=()):
    return dict(bboxes=torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),conf=torch.tensor(scores,dtype=torch.float32),cls=torch.zeros(len(scores)),extra=torch.zeros((len(scores),0)))

class FusionBehavior(unittest.TestCase):
    def test_two_supporting_views_rank_above_single_view_false_alarm(self):
        views=[view([[10,10,30,30],[70,10,90,30]],[.8,.9]),view([[12,10,32,30]],[.4])]
        out=fuse_views(views,100,50)
        self.assertEqual(len(out['conf']),2)
        torch.testing.assert_close(out['conf'],torch.tensor([.6,.45]))
        torch.testing.assert_close(out['bboxes'],torch.tensor([[10+2/3,10,30+2/3,30],[70,10,90,30]]))
        self.assertTrue(torch.equal(out['cls'],torch.zeros(2)))
        torch.testing.assert_close(views[0]['bboxes'],torch.tensor([[10.,10.,30.,30.],[70.,10.,90.,30.]]))

    def test_both_views_preserve_consistent_confidence(self):
        out=fuse_views([view([[10,10,30,30]],[.8]) for _ in range(2)],100,50)
        torch.testing.assert_close(out['conf'],torch.tensor([.8]))
        torch.testing.assert_close(out['bboxes'],torch.tensor([[10.,10.,30.,30.]]))

    def test_empty_views_return_valid_empty_detection_tensors(self):
        out=fuse_views([view() for _ in range(2)],100,50)
        self.assertEqual(out['bboxes'].shape,(0,4));self.assertEqual(out['conf'].shape,(0,));self.assertEqual(out['extra'].shape,(0,0))

    def test_clips_outside_box_to_non_square_frame(self):
        out=fuse_views([view([[-10,0,20,60]],[.8]),view()],100,50)
        torch.testing.assert_close(out['bboxes'],torch.tensor([[0.,0.,20.,50.]]))
        torch.testing.assert_close(out['conf'],torch.tensor([.4]))

    def test_native_validator_fuses_two_prediction_segments(self):
        validator=FlipFusionValidator(args=dict(task='detect',conf=.001,iou=.5,plots=False))
        validator.device=torch.device('cpu');validator.end2end=False;validator.nc=1
        validator.preprocess(dict(img=torch.zeros(1,3,64,96,dtype=torch.uint8)))
        raw=torch.tensor([[[20.,20.],[20.,20.],[20.,20.],[20.,20.],[.8,.4]]])
        before=raw.clone();out=validator.postprocess((raw,None))[0]
        torch.testing.assert_close(out['conf'],torch.tensor([.6]))
        torch.testing.assert_close(out['bboxes'],torch.tensor([[10.,10.,30.,30.]]))
        torch.testing.assert_close(raw,before,rtol=0,atol=0)
        self.assertEqual(validator.fused_images,1)

if __name__=='__main__':unittest.main()
