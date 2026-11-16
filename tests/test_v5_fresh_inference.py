import importlib.util
import copy
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires project inference environment')
class FreshInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import v5_fresh_inference as fresh
        cls.torch, cls.fresh = torch, fresh
        torch.set_num_threads(1)

    def test_validator_retains_two_class_labels_for_one_box(self):
        with tempfile.TemporaryDirectory() as directory:
            v = self.fresh.PredictionOnlyValidator(save_dir=Path(directory), args=dict(task='detect', conf=.001, iou=.7, max_det=300, plots=False))
            v.nc, v.end2end = 2, False
            result = v.postprocess(self.torch.tensor([[[10.], [10.], [4.], [4.], [.9], [.8]]]))
            self.assertEqual(sorted(result[0]['cls'].tolist()), [0, 1])

    def test_no_labels_yields_predictions_without_accuracy_metrics(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v = self.fresh.PredictionOnlyValidator(save_dir=root, args=dict(task='detect', plots=False, save_txt=True, save_conf=True))
            v.seen, v.names, v.device = 0, {0: 'food'}, torch.device('cpu')
            batch = dict(batch_idx=torch.empty(0), cls=torch.empty(0, 1), bboxes=torch.empty(0, 4),
                         ori_shape=[(100, 200)], img=torch.zeros(1, 3, 100, 200),
                         ratio_pad=[((1., 1.), (0., 0.))], im_file=['fresh.jpg'])
            pred = dict(bboxes=torch.tensor([[10., 10., 50., 30.]]), conf=torch.tensor([.8]), cls=torch.tensor([0.]), extra=torch.empty(1, 0))
            v.update_metrics([pred], batch)
            self.assertEqual(v.seen, 1)
            self.assertEqual(v.get_stats(), {})
            self.assertEqual(len((root/'labels/fresh.txt').read_text().splitlines()), 1)

    def test_routing_discards_unowned_parent_predictions(self):
        names = self.fresh.NAMES
        models = [dict(classes=[n for n in names if n != 'utensil']), dict(classes=['utensil'])]
        parent = {'fresh': [[.1, .1, .4, .4, .9, 0], [.2, .2, .4, .4, .7, 2]]}
        expert = {'fresh': [[.3, .3, .5, .5, .8, 2]]}
        result = self.fresh.assemble_predictions([parent, expert], models, {'fresh': []}, dict(class_id=5, min_iou=.55, min_confidence=.25, weight=.5))
        self.assertEqual(result['fresh'], [parent['fresh'][0], expert['fresh'][0]])
        with self.assertRaises(ValueError):
            self.fresh.assemble_predictions([parent, {'other': []}], models, {'fresh': []}, dict(class_id=5))

    def test_optional_recovery_routes_expert_and_preserves_primary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root/'model.pt'; weights.write_bytes(b'checkpoint fixture')
            picture = root/'fresh.jpg'; picture.write_bytes(b'mocked image')
            primary = dict(weights=str(weights), sha256=self.fresh.sha(weights),
                           classes=self.fresh.NAMES, source_classes=self.fresh.NAMES)
            parameters = dict(class_id=4, min_confidence=.65, max_area=.3,
                              base_confidence=.25, min_iou=.55, confidence_multiplier=.5)
            recovery = dict(weights=str(weights), sha256=self.fresh.sha(weights),
                            classes=['drink_carton'], source_classes=['drink_carton'],
                            settings=dict(imgsz=640), parameters=parameters,
                            implementation_sha256=self.fresh.sha(self.fresh.RECOVERY_SOURCE))
            manifest = dict(status='verified_development_candidate', models=[primary],
                            recovery_experts=[recovery],
                            settings=dict(imgsz=800, batch=1, device='cpu', conf=.001, iou=.7, max_det=300),
                            geometry_refinement=dict(class_id=5, min_iou=.55, min_confidence=.25, weight=.5,
                                implementation_sha256=self.fresh.sha(self.fresh.GEOMETRY_SOURCE),
                                expert=dict(checkpoint=str(weights), sha256=self.fresh.sha(weights))))
            path = root/'candidate.json'; path.write_text(json.dumps(manifest))
            inputs = [dict(name='fresh', image_path=str(picture), sha256=self.fresh.sha(picture))]
            base = {'fresh': [[.1,.1,.3,.4,.9,4], [.5,.5,.8,.8,.8,0]]}
            extra = {'fresh': [[.6,.1,.8,.4,.8,4]]}
            with patch.object(self.fresh, 'infer_native', side_effect=[base, {'fresh': []}, extra]) as infer:
                result = self.fresh.run_candidate(path, inputs, root/'out')
            self.assertEqual(result['fresh'], base['fresh'] + [[.6,.1,.8,.4,.4,4]])
            self.assertEqual(infer.call_args_list[2].kwargs['imgsz'], 640)
            receipt = json.loads((root/'out/receipt.json').read_text())
            self.assertEqual(receipt['recovery_settings'][0]['parameters'], parameters)
            wrapper_manifest = copy.deepcopy(manifest)
            wrapper = wrapper_manifest['recovery_experts'][0]
            wrapper.update(classes=['wrapper'], source_classes=['wrapper'])
            wrapper['parameters']['class_id'] = 5
            path.write_text(json.dumps(wrapper_manifest))
            raw = {'fresh': [[.6,.1,.8,.4,.8,5]]}
            geometry = {'fresh': [[.62,.1,.82,.4,.9,5]]}
            with patch.object(self.fresh, 'infer_native', side_effect=[base, geometry, raw]):
                result = self.fresh.run_candidate(path, inputs, root/'wrapper-out')
            expected = self.fresh.refine_geometry(raw['fresh'], geometry['fresh'],
                class_id=5, min_iou=.55, min_confidence=.25, weight=.5)[0]
            self.assertEqual(result['fresh'][-1], [*expected[:4], .4, 5])
            for changes in [dict(sha256='changed'), dict(implementation_sha256='changed'),
                            dict(settings=dict(imgsz=641)), dict(source_classes=['wrapper']),
                            dict(parameters=parameters | dict(class_id=5)),
                            dict(parameters=parameters | dict(min_confidence=float('nan')))]:
                invalid = copy.deepcopy(manifest)
                invalid['recovery_experts'][0].update(changes)
                path.write_text(json.dumps(invalid))
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    self.fresh.load_candidate(path)
            for experts in [None, [recovery, recovery]]:
                invalid = manifest | dict(recovery_experts=experts)
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError): self.fresh.load_candidate(path)

    def test_native_resolution_accepts_only_explicit_image_size(self):
        self.assertEqual(self.fresh.native_resolution({}), 800)
        self.assertEqual(self.fresh.native_resolution(dict(settings=dict(imgsz=640))), 640)
        for settings in [dict(imgsz=0), dict(imgsz=641), dict(imgsz=True),
                         dict(imgsz="640"), dict(conf=.25), None]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.fresh.native_resolution(dict(settings=settings))

    def test_candidate_resolution_reaches_inference_and_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root/'model.pt'; weights.write_bytes(b'identity fixture')
            picture = root/'fresh.jpg'; picture.write_bytes(b'mocked image fixture')
            model = dict(weights=str(weights), sha256=self.fresh.sha(weights),
                         classes=self.fresh.NAMES, source_classes=self.fresh.NAMES,
                         settings=dict(imgsz=640))
            manifest = dict(status='verified_development_candidate', models=[model],
                            settings=dict(imgsz=800, batch=1, device='cpu', conf=.001, iou=.7, max_det=300),
                            geometry_refinement=dict(class_id=5, min_iou=.55, min_confidence=.25, weight=.5,
                                implementation_sha256=self.fresh.sha(self.fresh.GEOMETRY_SOURCE),
                                expert=dict(checkpoint=str(weights), sha256=self.fresh.sha(weights))))
            path = root/'candidate.json'; path.write_text(json.dumps(manifest))
            inputs = [dict(name='fresh', image_path=str(picture), sha256=self.fresh.sha(picture))]
            with patch.object(self.fresh, 'infer_native', side_effect=[{'fresh': []}, {'fresh': []}]) as infer:
                self.fresh.run_candidate(path, inputs, root/'output')
            self.assertEqual(infer.call_args_list[0].kwargs['imgsz'], 640)
            self.assertEqual(infer.call_args_list[1].kwargs.get('imgsz', 800), 800)
            receipt = json.loads((root/'output/receipt.json').read_text())
            self.assertEqual(receipt['model_settings'][0]['imgsz'], 640)
            model['settings'] = dict(conf=.25); path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): self.fresh.load_candidate(path)

    def test_candidate_checks_all_ownership_and_weight_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root/'model.pt'; weights.write_bytes(b'fake checkpoint for identity test')
            manifest = dict(status='verified_development_candidate', models=[dict(weights=str(weights), sha256=self.fresh.sha(weights), classes=self.fresh.NAMES, source_classes=self.fresh.NAMES)],
                            settings=dict(imgsz=800, batch=1, device='cpu', conf=.001, iou=.7, max_det=300),
                            geometry_refinement=dict(class_id=5, min_iou=.55, min_confidence=.25, weight=.5,
                                implementation_sha256=self.fresh.sha(self.fresh.GEOMETRY_SOURCE), expert=dict(checkpoint=str(weights), sha256=self.fresh.sha(weights))))
            path = root/'candidate.json';path.write_text(json.dumps(manifest))
            self.fresh.load_candidate(path)
            manifest['models'].append(dict(manifest['models'][0]));path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):self.fresh.load_candidate(path)
            manifest['models'].pop();path.write_text(json.dumps(manifest));weights.write_bytes(b'changed')
            with self.assertRaises(ValueError):self.fresh.load_candidate(path)


if __name__ == '__main__':
    unittest.main()
