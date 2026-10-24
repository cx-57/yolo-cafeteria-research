import importlib.util
from pathlib import Path
import unittest
import torch

torch.set_num_threads(1)
spec=importlib.util.spec_from_file_location('v31_trainability',Path(__file__).with_name('trainability.py'));guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)

class TrainabilityTests(unittest.TestCase):
    def model(self):
        model=torch.nn.Module();model.model=torch.nn.Sequential(*[torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.BatchNorm1d(2)) for _ in range(24)])
        for name,p in model.named_parameters():p.requires_grad_(int(name.split('.')[1])>=11)
        for layer in model.modules():
            if isinstance(layer,torch.nn.BatchNorm1d):layer.eval()
        return model

    def test_actual_parameters_and_bn_guard_backbone_neck_and_head(self):
        model=self.model();report=guard.audit(model)
        self.assertEqual(report['trainable_layer_indices'],list(range(11,24)))
        self.assertTrue(report['backbone_parameters_fixed'])
        self.assertTrue(report['all_BatchNorm_statistics_fixed'])
        self.assertEqual(report['normalization_modules'],24)
        next(model.model[10].parameters()).requires_grad_(True)
        with self.assertRaises(AssertionError):guard.audit(model)

    def test_accidental_head_only_or_train_mode_bn_rejected(self):
        model=self.model();model.model[10][1].train()
        with self.assertRaises(AssertionError):guard.audit(model)
        model=self.model()
        for name,p in model.named_parameters():p.requires_grad_(name.startswith('model.23.'))
        with self.assertRaises(AssertionError):guard.audit(model)

if __name__=='__main__':unittest.main()
