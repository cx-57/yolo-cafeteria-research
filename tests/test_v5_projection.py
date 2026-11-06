import copy
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    import v5_projection as p
except ImportError:
    p = None


@unittest.skipUnless(importlib.util.find_spec('ultralytics'), 'Ultralytics unavailable')
class ProjectionTests(unittest.TestCase):
    def test_refinement_requires_gain_without_large_class_regressions(self):
        self.assertIsNotNone(p)
        self.assertTrue(hasattr(p, 'assess_refinement'), 'comparison gate is missing')
        from v5_dataset import NAMES
        base = {n: {'AP50': .7, 'AP50_95': .4} for n in NAMES}
        candidate = copy.deepcopy(base)
        candidate['utensil']['AP50_95'] = .48
        self.assertTrue(p.assess_refinement(base, candidate)['accept'])
        candidate['plate']['AP50_95'] = .3
        self.assertFalse(p.assess_refinement(base, candidate)['accept'])
        self.assertFalse(p.assess_refinement(base, base)['accept'])
        candidate['utensil']['AP50_95'] = float('nan')
        with self.assertRaises(ValueError): p.assess_refinement(base, candidate)

    def test_removing_bottle_preserves_learned_boxes_and_six_class_scores(self):
        self.assertIsNotNone(p, 'weight-preserving projection is missing')
        import torch
        from ultralytics.nn.tasks import DetectionModel
        from v5_dataset import NAMES
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        source = DetectionModel('yolo11n.yaml', nc=7, verbose=False).eval()
        source.names = dict(enumerate(NAMES + ['bottle']))
        before_state = copy.deepcopy(source.state_dict())
        image = torch.randn(1, 3, 64, 64)
        with torch.inference_mode():
            before = source(image)[0]
        projected = p.without_bottle(source)
        with torch.inference_mode():
            after = projected(image)[0]
        self.assertEqual(after.shape[1], 10)
        torch.testing.assert_close(after[:, :4], before[:, :4])
        torch.testing.assert_close(after[:, 4:], before[:, 4:10])
        self.assertEqual(projected.names, dict(enumerate(NAMES)))
        self.assertEqual(source.model[-1].nc, 7)
        for name, value in before_state.items():
            torch.testing.assert_close(source.state_dict()[name], value)

    def test_wrong_class_order_is_rejected(self):
        self.assertIsNotNone(p, 'weight-preserving projection is missing')
        from types import SimpleNamespace
        model = SimpleNamespace(names={0: 'bottle', 1: 'food'})
        with self.assertRaisesRegex(ValueError, 'class'):
            p.without_bottle(model)


if __name__ == '__main__': unittest.main()
