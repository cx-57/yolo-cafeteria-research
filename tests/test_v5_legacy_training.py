import importlib.util
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires the project Torch environment')
class LegacyTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from v5_legacy_training import preserve_utensil_model
        cls.torch = torch
        cls.preserve = staticmethod(preserve_utensil_model)
        torch.set_num_threads(1)

    def model(self):
        nn = self.torch.nn
        source = nn.Module()
        head = nn.Module()
        head.nc = 1
        head.cv2 = nn.ModuleList([nn.Conv2d(4, 64, 1)])
        head.cv3 = nn.ModuleList([nn.Sequential(nn.Conv2d(4, 3, 1), nn.MaxPool3d((3, 1, 1)))])
        source.model = nn.ModuleList([nn.Conv2d(3, 4, 1), head])
        source.names = {0: 'utensil'}
        return source

    def test_custom_pool_and_outputs_survive_without_source_mutation(self):
        source = self.model()
        clone = self.preserve(source)
        x = self.torch.randn(1, 4, 8, 8)
        self.torch.testing.assert_close(source.model[-1].cv3[0](x), clone.model[-1].cv3[0](x), rtol=0, atol=0)
        self.assertIsInstance(clone.model[-1].cv3[0][-1], self.torch.nn.MaxPool3d)
        self.assertEqual(set(source.state_dict()), set(clone.state_dict()))
        with self.torch.no_grad():
            next(clone.parameters()).add_(1)
        self.assertFalse(self.torch.equal(next(source.parameters()), next(clone.parameters())))

    def test_head_backpropagation_and_standard_checkpoint_roundtrip(self):
        clone = self.preserve(self.model())
        for p in clone.parameters():
            p.requires_grad_(False)
        for p in clone.model[-1].cv3.parameters():
            p.requires_grad_(True)
        x = self.torch.randn(1, 4, 8, 8)
        output = clone.model[-1].cv3[0](x)
        self.assertEqual(tuple(output.shape), (1, 1, 8, 8))
        output.square().mean().backward()
        self.assertGreater(clone.model[-1].cv3[0][0].weight.grad.abs().sum().item(), 0)
        self.assertIsNone(clone.model[0].weight.grad)
        buffer = io.BytesIO()
        self.torch.save(clone, buffer)
        buffer.seek(0)
        restored = self.torch.load(buffer, weights_only=False)
        self.torch.testing.assert_close(output, restored.model[-1].cv3[0](x), rtol=0, atol=0)

    def test_wrong_classes_and_missing_custom_head_are_rejected(self):
        source = self.model()
        source.names = {0: 'wrapper'}
        with self.assertRaises(ValueError):
            self.preserve(source)
        source = self.model()
        source.model[-1].cv3[0][-1] = self.torch.nn.Identity()
        with self.assertRaises(ValueError):
            self.preserve(source)


if __name__ == '__main__':
    unittest.main()
