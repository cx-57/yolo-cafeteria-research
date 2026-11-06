"""Explicitly requested specialist trials for the six-class V5 system."""
from collections import Counter
import gc
import importlib.util
import json
import math
import os
from pathlib import Path

import yaml

from v5_dataset import NAMES, sha, strip_bottle, verify_dataset, write_json
from specialists import score_predictions, validate_prediction_rows


def specialist_labels(text, global_ids):
    if not global_ids or len(set(global_ids)) != len(global_ids) or not set(global_ids) <= set(range(6)):
        raise ValueError('Invalid specialist class ownership')
    clean, removed = strip_bottle(text)
    if removed:
        raise ValueError('Bottle is outside the six-class specialist task')
    mapping = {c: i for i, c in enumerate(global_ids)}
    result = []
    for line in clean.splitlines():
        parts = line.split()
        if int(parts[0]) in mapping:
            parts[0] = str(mapping[int(parts[0])])
            result.append(' '.join(parts))
    return '\n'.join(result) + ('\n' if result else '')


def training_subset(records, global_ids):
    train = [r for r in records if r['split'] == 'train']
    positives, negatives = [], []
    for r in train:
        positive = any(int(c) in global_ids and count > 0 for c, count in r['counts'].items())
        (positives if positive else negatives).append(r)
    if not positives:
        raise ValueError('No target-positive training images')
    # Deterministic, up to one negative per positive. Never mine held-out photos.
    negatives.sort(key=lambda r: (r['image_sha256'], r['image_path']))
    return positives + negatives[:len(positives)]


def merge_owned_predictions(base, experts):
    owned = set()
    for expert in experts:
        names = expert['classes']
        if (not names or not set(names) <= set(NAMES) or len(set(names)) != len(names)
                or owned.intersection(names) or set(expert['predictions']) != set(base)):
            raise ValueError('Duplicate ownership, invalid classes or mismatched images')
        owned.update(names)
        ids = {NAMES.index(n) for n in names}
        for rows in expert['predictions'].values():
            validate_prediction_rows(rows)
            if any(int(row[5]) not in ids for row in rows):
                raise ValueError('Specialist predicted a class it does not own')
    drop = {NAMES.index(n) for n in owned}
    result = {}
    for name, rows in base.items():
        validate_prediction_rows(rows)
        if any(int(r[5]) >= 6 for r in rows):
            raise ValueError('Bottle cannot appear in the six-class pipeline')
        result[name] = [list(r) for r in rows if int(r[5]) not in drop]
        for expert in experts:
            result[name].extend(list(r) for r in expert['predictions'][name])
    return result


def accept_specialist(baseline, candidate, owned):
    for result in (baseline, candidate):
        for n in owned:
            for key in ('AP50', 'AP50_95'):
                value = result[n][key]
                if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError('Invalid specialist score')
    return all(candidate[n]['AP50_95'] >= baseline[n]['AP50_95'] + .01 - 1e-12
               and candidate[n]['AP50'] >= baseline[n]['AP50'] - .02 - 1e-12 for n in owned)


def validation_records(dataset, receipt=None):
    dataset = Path(dataset)
    receipt = receipt or verify_dataset(dataset)
    return [dict(name=Path(r['image_path']).stem,
                 rows=(dataset / r['label_path']).read_text().splitlines())
            for r in receipt['images'] if r['split'] == 'val']


def build_view(dataset, output, global_ids, include_train=True):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    receipt = verify_dataset(dataset)
    if output.exists():
        raise FileExistsError(output)
    if not global_ids or len(set(global_ids)) != len(global_ids) or not set(global_ids) <= set(range(6)):
        raise ValueError('Invalid specialist classes')
    records = training_subset(receipt['images'], global_ids) if include_train else []
    records += [r for r in receipt['images'] if r['split'] == 'val']
    output.mkdir(parents=True)
    for split in ('train', 'val'):
        (output / 'images' / split).mkdir(parents=True)
        (output / 'labels' / split).mkdir(parents=True)
    files, counts = {}, Counter()
    for r in records:
        image = output / r['image_path']
        label = output / r['label_path']
        os.link(dataset / r['image_path'], image)
        text = specialist_labels((dataset / r['label_path']).read_text(), global_ids)
        label.write_text(text)
        files[r['image_path']] = r['image_sha256']
        files[r['label_path']] = sha(label)
        counts[r['split'] + '_images'] += 1
        counts[r['split'] + '_boxes'] += len(text.splitlines())
    cfg = dict(path=str(output), train='images/train', val='images/val',
               nc=len(global_ids), names=[NAMES[i] for i in global_ids])
    (output / 'data.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    files['data.yaml'] = sha(output / 'data.yaml')
    report = dict(global_ids=list(global_ids), source_dataset=str(dataset),
                  source_receipt_sha256=sha(dataset / 'dataset_receipt.json'),
                  files=files, counts=dict(counts), test_images_used=0,
                  annotation_status=receipt['annotation_status'])
    write_json(output / 'view.json', report)
    return report


def verify_view(output):
    output = Path(output)
    receipt = json.loads((output / 'view.json').read_text())
    source = Path(receipt['source_dataset'])
    verify_dataset(source)
    if sha(source / 'dataset_receipt.json') != receipt['source_receipt_sha256']:
        raise ValueError('Source dataset receipt changed')
    for rel, expected in receipt['files'].items():
        p = output / rel
        if not p.is_file() or sha(p) != expected:
            raise ValueError('Changed specialist dataset file: ' + rel)
    actual = {str(p.relative_to(output)) for p in (output / 'images').rglob('*') if p.is_file()}
    expected = {p for p in receipt['files'] if p.startswith('images/')}
    if actual != expected:
        raise ValueError('Extra or missing specialist images')
    return receipt


def read_predictions(folder, records, native_to_global):
    folder = Path(folder)
    result = {}
    for record in records:
        rows = []
        path = folder / (record['name'] + '.txt')
        for line in path.read_text().splitlines() if path.is_file() else []:
            parts = line.split()
            if len(parts) != 6:
                raise ValueError('Expected native class/xywh/confidence predictions')
            c, x, y, w, h, confidence = map(float, parts)
            if c != int(c) or int(c) not in native_to_global:
                raise ValueError('Unexpected native specialist prediction class')
            rows.append([max(0, x-w/2), max(0, y-h/2), min(1, x+w/2), min(1, y+h/2),
                         confidence, native_to_global[int(c)]])
        validate_prediction_rows(rows)
        result[record['name']] = rows
    return result


def evaluate_checkpoint(checkpoint, view, output, device='mps', legacy=False):
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.val import DetectionValidator
    output, view, checkpoint = Path(output).resolve(), Path(view).resolve(), Path(checkpoint).resolve()
    if output.exists():
        raise FileExistsError(output)
    receipt = verify_view(view)
    ids = receipt['global_ids']
    records = validation_records(receipt['source_dataset'])
    model = YOLO(str(checkpoint))
    if list(model.names.values()) != [NAMES[i] for i in ids]:
        raise ValueError('Checkpoint native names differ from specialist label mapping')
    if legacy:
        path = Path(__file__).resolve().parents[1] / 'experiments/yolo-v4-4class/v4_hybrid_inference.py'
        spec = importlib.util.spec_from_file_location('v5_legacy_adapter', path)
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        adapter.repair_legacy_head_pooling(model)
    for p in (view / 'labels').glob('*.cache'):
        p.unlink()
    observed = []
    class TrackedValidator(DetectionValidator):
        def __call__(self, *args, **kwargs):
            result = super().__call__(*args, **kwargs)
            observed.append(self.seen)
            return result
    model.val(validator=TrackedValidator, data=str(view / 'data.yaml'), split='val', imgsz=800, batch=1, device=device,
              workers=0, conf=.001, iou=.7, max_det=300, plots=False, save_txt=True,
              save_conf=True, project=str(output.parent), name=output.name, exist_ok=False)
    if observed != [len(records)]:
        raise ValueError('Specialist validation skipped or added images')
    predictions = read_predictions(output / 'labels', records, dict(enumerate(ids)))
    scores = score_predictions(records, predictions, set(ids))
    write_json(output / 'predictions.json', predictions)
    write_json(output / 'report.json', dict(scores=scores, checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint), source_receipt_sha256=receipt['source_receipt_sha256'],
        view_sha256=sha(view / 'view.json'), predictions_sha256=sha(output / 'predictions.json'),
        global_ids=ids, device=device, imgsz=800, test_evaluated=False))
    del model
    gc.collect()
    if device == 'mps':
        torch.mps.empty_cache()
    return scores, predictions


def load_evaluation(output, checkpoint, view):
    output, checkpoint, view = Path(output), Path(checkpoint), Path(view)
    receipt = verify_view(view)
    report = json.loads((output / 'report.json').read_text())
    if (report['checkpoint_sha256'] != sha(checkpoint)
            or report['source_receipt_sha256'] != receipt['source_receipt_sha256']
            or report['view_sha256'] != sha(view / 'view.json')
            or report['predictions_sha256'] != sha(output / 'predictions.json')):
        raise ValueError('Changed evaluation checkpoint, dataset or prediction receipt')
    predictions = json.loads((output / 'predictions.json').read_text())
    records = validation_records(receipt['source_dataset'])
    if set(predictions) != {r['name'] for r in records}:
        raise ValueError('Evaluation prediction cohort is incomplete')
    for rows in predictions.values():
        validate_prediction_rows(rows)
    scores = score_predictions(records, predictions, set(receipt['global_ids']))
    return scores, predictions
