"""Unscored image inference with the exact V5 validation NMS and owner policy."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path

import yaml
from ultralytics.models.yolo.detect.val import DetectionValidator

from v5_dataset import NAMES, sha, write_json
from v5_specialists import merge_owned_predictions, read_predictions
from v5_box_refinement import refine_geometry
from v5_recovery import recover_detections

GEOMETRY_SOURCE = Path(__file__).with_name('v5_box_refinement.py')
RECOVERY_SOURCE = Path(__file__).with_name('v5_recovery.py')


def native_resolution(model):
    settings = model.get('settings', {})
    if not isinstance(settings, dict) or set(settings) - {'imgsz'}:
        raise ValueError('A detector may override image resolution only')
    size = settings.get('imgsz', 800)
    if type(size) is not int or size <= 0 or size % 32:
        raise ValueError('Image resolution must be a positive multiple of32')
    return size


class PredictionOnlyValidator(DetectionValidator):
    """Keep validation preprocessing/NMS; never interpret absent truth as accuracy."""
    def update_metrics(self, preds, batch):
        for index, pred in enumerate(preds):
            self.seen += 1
            prepared = self._prepare_batch(index, batch)
            if len(pred['cls']):
                self.save_one_txt(self.scale_preds(pred, prepared), True, prepared['ori_shape'],
                                  self.save_dir/'labels'/f"{Path(prepared['im_file']).stem}.txt")

    def get_stats(self):
        return {}

    def finalize_metrics(self):
        pass

    def print_results(self):
        pass


def load_candidate(path):
    candidate = json.loads(Path(path).read_text())
    if candidate.get('status') != 'verified_development_candidate':
        raise ValueError('Expected a verified candidate manifest')
    models = candidate['models']
    if sorted(n for m in models for n in m['classes']) != sorted(NAMES):
        raise ValueError('Exactly one owner is required for each of the six classes')
    for model in models:
        native_resolution(model)
        native = model['source_classes']
        if len(set(native)) != len(native) or not set(model['classes']) <= set(native) <= set(NAMES):
            raise ValueError('Invalid native class mapping')
        if sha(model['weights']) != model['sha256']:
            raise ValueError('Checkpoint hash changed')
    geometry = candidate['geometry_refinement']
    if geometry['class_id'] != 5 or geometry['implementation_sha256'] != sha(GEOMETRY_SOURCE):
        raise ValueError('Geometry policy or implementation changed')
    # Reuse the helper's parameter validation without modifying a prediction.
    refine_geometry([], [], **{k: geometry[k] for k in ('class_id', 'min_iou', 'min_confidence', 'weight')})
    if sha(geometry['expert']['checkpoint']) != geometry['expert']['sha256']:
        raise ValueError('Geometry checkpoint hash changed')
    recovery = candidate.get('recovery_experts', [])
    if not isinstance(recovery, list):
        raise ValueError('Recovery experts must be a list')
    recovered_classes = set()
    for expert in recovery:
        native_resolution(expert)
        native = expert['source_classes']
        owned = expert['classes']
        parameters = expert['parameters']
        if (not isinstance(native, list) or len(set(native)) != len(native)
                or not set(native) <= set(NAMES)
                or len(owned) != 1 or owned[0] not in native
                or owned[0] in recovered_classes
                or set(parameters) != {'class_id', 'min_confidence', 'max_area',
                                       'base_confidence', 'min_iou', 'confidence_multiplier'}
                or parameters['class_id'] != NAMES.index(owned[0])):
            raise ValueError('Invalid recovery class mapping or parameters')
        recover_detections([], [], **parameters)
        if expert['implementation_sha256'] != sha(RECOVERY_SOURCE):
            raise ValueError('Recovery implementation changed')
        if sha(expert['weights']) != expert['sha256']:
            raise ValueError('Recovery checkpoint hash changed')
        recovered_classes.add(owned[0])
    settings = candidate['settings']
    expected = dict(imgsz=800, batch=1, device='cpu', conf=.001, iou=.7, max_det=300)
    if settings != expected:
        raise ValueError('Candidate settings differ from the fixed validation protocol')
    return candidate


def assemble_predictions(outputs, models, geometry_output, geometry_settings):
    if not outputs or len(outputs) != len(models):
        raise ValueError('Model output count differs')
    cohort = set(outputs[0])
    if not cohort or set(geometry_output) != cohort or any(set(o) != cohort for o in outputs):
        raise ValueError('Prediction cohorts differ')
    experts = []
    for output, model in zip(outputs, models):
        owned = {NAMES.index(n) for n in model['classes']}
        filtered = {key: [row for row in rows if int(row[5]) in owned] for key, rows in output.items()}
        experts.append(dict(classes=model['classes'], predictions=filtered))
    merged = merge_owned_predictions({key: [] for key in outputs[0]}, experts)
    return {key: refine_geometry(rows, geometry_output[key], **geometry_settings) for key, rows in merged.items()}


def infer_native(checkpoint, native_names, images, output, legacy=False, imgsz=800):
    """No truth labels or AP computation; emit the same normalized TXT protocol."""
    from ultralytics import YOLO
    imgsz = native_resolution(dict(settings=dict(imgsz=imgsz)))
    output, checkpoint = Path(output), Path(checkpoint)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    before = sha(checkpoint)
    seen_names = set()
    for item in images:
        path = Path(item['image_path'])
        if item['name'] != path.stem or item['name'] in seen_names or sha(path) != item['sha256']:
            raise ValueError('Changed, duplicate or invalid input identity')
        seen_names.add(item['name'])
    view = output/'native-view'
    (view/'images/train').mkdir(parents=True)
    (view/'images/val').mkdir()
    for item in images:
        path = Path(item['image_path'])
        os.link(path, view/'images/val'/path.name)
    data = dict(path=str(view.resolve()), train='images/train', val='images/val', nc=len(native_names), names=native_names)
    (view/'data.yaml').write_text(yaml.safe_dump(data, sort_keys=False))
    model = YOLO(str(checkpoint))
    if [model.names[i] for i in range(len(model.names))] != native_names:
        raise ValueError('Native checkpoint names differ')
    if legacy:
        path = Path(__file__).resolve().parents[1]/'experiments/yolo-v4-4class/v4_hybrid_inference.py'
        spec = importlib.util.spec_from_file_location('v5_fresh_legacy', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.repair_legacy_head_pooling(model)
    observed = []
    class Tracked(PredictionOnlyValidator):
        def __call__(self, *args, **kwargs):
            result = super().__call__(*args, **kwargs)
            observed.append(self.seen)
            return result
    model.val(validator=Tracked, data=str(view/'data.yaml'), split='val', imgsz=imgsz, batch=1,
              device='cpu', workers=0, conf=.001, iou=.7, max_det=300, augment=False, half=False,
              channels_last=False, plots=False, save_txt=True, save_conf=True,
              project=str(output), name='inference', exist_ok=False)
    if observed != [len(images)]:
        raise ValueError('Inference did not process the complete cohort')
    result = read_predictions(output/'inference/labels', images, {i: NAMES.index(n) for i, n in enumerate(native_names)})
    if sha(checkpoint) != before or any(sha(i['image_path']) != i['sha256'] for i in images):
        raise ValueError('Checkpoint or inputs changed during inference')
    write_json(output/'predictions.json', result)
    write_json(output/'receipt.json', dict(checkpoint=str(checkpoint), checkpoint_sha256=before,
        input_images=images, images_processed=len(images), native_names=native_names,
        predictions_sha256=sha(output/'predictions.json'), labels_status='unknown', accuracy_measured=False,
        settings=dict(imgsz=imgsz, batch=1, device='cpu', conf=.001, iou=.7, max_det=300, multi_label=True)))
    del model
    gc.collect()
    return result


def run_candidate(candidate_path, images, output):
    import torch
    torch.set_num_threads(1)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    candidate_sha256 = sha(candidate_path)
    candidate = load_candidate(candidate_path)
    if not images or len({i['name'] for i in images}) != len(images) or len({i['sha256'] for i in images}) != len(images):
        raise ValueError('Input images must be nonempty and unique')
    output.mkdir(parents=True)
    outputs = []
    for index, model in enumerate(candidate['models']):
        print('Running owner', index, model['classes'], flush=True)
        outputs.append(infer_native(model['weights'], model['source_classes'], images, output/f'owner-{index}',
                                    model.get('legacy_adapter', False), imgsz=native_resolution(model)))
    policy = candidate['geometry_refinement']
    expert = infer_native(policy['expert']['checkpoint'], ['wrapper'], images, output/'wrapper-geometry')
    parameters = {key: policy[key] for key in ('class_id', 'min_iou', 'min_confidence', 'weight')}
    result = assemble_predictions(outputs, candidate['models'], expert, parameters)
    for index, recovery in enumerate(candidate.get('recovery_experts', [])):
        extra = infer_native(recovery['weights'], recovery['source_classes'], images,
                             output/f'recovery-{index}', recovery.get('legacy_adapter', False),
                             imgsz=native_resolution(recovery))
        if set(extra) != set(result):
            raise ValueError('Recovery prediction cohort differs')
        if recovery['parameters']['class_id'] == policy['class_id']:
            extra = {name: refine_geometry(rows, expert[name], **parameters)
                     for name, rows in extra.items()}
        result = {name: recover_detections(rows, extra[name], **recovery['parameters'])
                  for name, rows in result.items()}
    write_json(output/'predictions.json', result)
    baseline_index = next((i for i, model in enumerate(candidate['models'])
                           if model['sha256'] == '6f2131e17ab93ce5df911b503fe5065f0450c164508ff83e4857188f74f453f6'), None)
    parent_sha256 = None
    if baseline_index is not None:
        write_json(output/'parent-v5-predictions.json', outputs[baseline_index])
        parent_sha256 = sha(output/'parent-v5-predictions.json')
    if sha(candidate_path) != candidate_sha256 or any(sha(i['image_path']) != i['sha256'] for i in images):
        raise ValueError('Candidate manifest or inputs changed during inference')
    load_candidate(candidate_path)
    write_json(output/'receipt.json', dict(candidate_path=str(candidate_path), candidate_sha256=candidate_sha256,
        implementation_sha256=sha(Path(__file__)), input_images=images, images_processed=len(images),
        predictions_sha256=sha(output/'predictions.json'), parent_predictions_sha256=parent_sha256,
        accuracy_measured=False, labels_status='unknown', settings=candidate['settings'],
        model_settings=[dict(classes=m['classes'], imgsz=native_resolution(m)) for m in candidate['models']],
        recovery_settings=[dict(classes=m['classes'], imgsz=native_resolution(m),
                                parameters=m['parameters'], sha256=m['sha256'],
                                implementation_sha256=m['implementation_sha256'])
                           for m in candidate.get('recovery_experts', [])],
        limitation='Unscored external visual evaluation; historical pretraining exposure is not certified absent'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--images', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'))
    images = [dict(name=p.stem, image_path=str(p.resolve()), sha256=sha(p)) for p in paths]
    run_candidate(args.candidate, images, args.output)


if __name__ == '__main__':
    main()
