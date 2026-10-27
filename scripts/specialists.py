"""Five frozen detectors, one eight-class output and common development evaluation."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path

from storage import ROOT, EVALUATION, resolve_image
NAMES = ['food', 'plate', 'utensil', 'hinged_container', 'drink_carton', 'wrapper', 'bottle', 'condiment_packet']
SETTINGS = dict(imgsz=640, batch=1, augment=False, conf=.001, iou=.7, max_det=300, half=False)
CONFIG = ROOT / 'configs/specialists.json'


def inference_settings(config):
    return dict(SETTINGS, **config.get('settings', {}))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def map_boxes(rows, source_names, owned_names):
    if isinstance(source_names, dict):
        source_names = [source_names[i] for i in range(len(source_names))]
    if set(source_names) != set(owned_names):
        raise ValueError(f'Checkpoint class names {source_names} differ from {owned_names}')
    result = []
    for row in rows:
        cls = int(row[5])
        if 0 <= cls < len(source_names):
            result.append([*map(float, row[:5]), NAMES.index(source_names[cls])])
    return result


def validate_annotations(records, classes):
    seen = set()
    for record in records:
        if record['name'] in seen:
            raise ValueError('Duplicate image name')
        seen.add(record['name'])
        if not set(classes).issubset(record.get('known_classes', [])):
            raise ValueError('Incomplete annotation coverage: ' + record['name'])
        for row in record['rows']:
            values = list(map(float, row.split()))
            if len(values) != 5 or not all(math.isfinite(v) for v in values):
                raise ValueError('Invalid YOLO annotation')
            c, x, y, w, h = values
            if c != int(c) or not 0 <= c < len(NAMES) or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError('Invalid class or box')


def verify_images(records):
    seen = set()
    for record in records:
        path = resolve_image(record['image_path'])
        if sha(path) != record['sha256']:
            raise ValueError('Changed image: ' + str(path))
        if record['sha256'] in seen:
            raise ValueError('Duplicate image bytes')
        seen.add(record['sha256'])


def validate_prediction_rows(rows):
    for row in rows:
        if len(row) != 6 or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in row):
            raise ValueError('Prediction requires six finite numbers')
        x1, y1, x2, y2, confidence, cls = row
        if not (0 <= x1 <= x2 <= 1 and 0 <= y1 <= y2 <= 1 and 0 <= confidence <= 1):
            raise ValueError('Invalid normalized prediction box or confidence')
        if cls != int(cls) or not 0 <= cls < len(NAMES):
            raise ValueError('Invalid prediction class')


def training_collisions(records, training):
    training = [r for r in training if r.get('split', 'train') == 'train']
    hashes = {r.get('sha256', r.get('sha', r.get('image_sha'))) for r in training} - {None}
    pixels = {r.get('pixels') for r in training} - {None}
    raw_pixels = {r.get('pixels_raw') for r in training} - {None}
    groups = {r.get('group') for r in training} - {None, ''}
    captures = {r.get('capture') for r in training} - {None, ''}
    return {r['name'] for r in records if r['sha256'] in hashes or
            r.get('pixels', '') in pixels or r.get('pixels_raw', '') in raw_pixels or r.get('group', '') in groups or
            r.get('capture', '') in captures}


def validate_receipt(value, records, config, config_sha):
    receipt = value.get('prediction_receipt', value)
    expected = dict(settings=inference_settings(config), config_sha256=config_sha, names=NAMES,
                    models=config['models'], box_format='normalized_xyxy_conf_class')
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError('Prediction configuration or checkpoint provenance differs from frozen protocol')
    by_sha = {r['sha256']: r['name'] for r in receipt['images']}
    if len(by_sha) != len(receipt['images']) or set(by_sha) != {r['sha256'] for r in records}:
        raise ValueError('Prediction images differ from annotation cohort')
    if set(receipt['predictions']) != set(by_sha.values()):
        raise ValueError('Missing or extra prediction image')
    predictions = {}
    for record in records:
        rows = receipt['predictions'][by_sha[record['sha256']]]
        validate_prediction_rows(rows)
        predictions[record['name']] = rows
    return dict(receipt, images=[{k: r[k] for k in ('name', 'sha256')} for r in records],
                predictions=predictions)


def predict(records, device='cpu', config_path=CONFIG):
    import torch
    import ultralytics
    from ultralytics import YOLO
    torch.set_num_threads(2)
    config = read(config_path)
    settings = inference_settings(config)
    models = config['models']
    if sorted(n for m in models for n in m['classes']) != sorted(NAMES):
        raise ValueError('Each output class must have exactly one owning detector')
    for spec in models:
        if sha(ROOT / spec['weights']) != spec['sha256']:
            raise ValueError('Changed checkpoint: ' + spec['weights'])
    records = [dict(r, image_path=str(resolve_image(r['image_path']))) for r in records]
    verify_images(records)
    predictions = {r['name']: [] for r in records}
    for spec in models:
        print('Loading ' + spec['id'], flush=True)
        model = YOLO(str(ROOT / spec['weights']))
        map_boxes([], model.names, spec['classes'])
        for index, record in enumerate(records):
            result = model.predict(record['image_path'], device=device, workers=0,
                                   verbose=False, **settings)[0]
            rows = map_boxes(result.boxes.data.cpu().tolist(), model.names, spec['classes'])
            height, width = result.orig_shape
            for row in rows:
                row[:4] = [row[0] / width, row[1] / height, row[2] / width, row[3] / height]
            predictions[record['name']].extend(rows)
            if (index + 1) % 25 == 0:
                print(f"{spec['id']}: {index + 1}/{len(records)}", flush=True)
        del result, model
        gc.collect()
        if device == 'mps':
            torch.mps.empty_cache()
    return dict(names=NAMES, settings=settings, device=device, config_sha256=sha(config_path),
                runtime=dict(torch=torch.__version__, ultralytics=ultralytics.__version__),
                models=models, images=[{k: r[k] for k in ('name', 'sha256')} for r in records],
                box_format='normalized_xyxy_conf_class', predictions=predictions)


def score_predictions(records, predictions, classes):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import ap_per_class, box_iou
    all_tp, all_conf, all_cls, all_targets = [], [], [], []
    matcher = SimpleNamespace(iouv=torch.linspace(.5, .95, 10))
    for record in records:
        if record['name'] not in predictions:
            raise ValueError('Missing prediction for ' + record['name'])
        target = []
        for row in record['rows']:
            c, x, y, w, h = map(float, row.split())
            if int(c) in classes:
                target.append([x-w/2, y-h/2, x+w/2, y+h/2, c])
        truth = torch.tensor(target, dtype=torch.float32).reshape(-1, 5)
        rows = [r for r in predictions[record['name']] if int(r[5]) in classes]
        pred = torch.tensor(rows, dtype=torch.float32).reshape(-1, 6)
        tp = BaseValidator.match_predictions(matcher, pred[:, 5], truth[:, 4],
                                             box_iou(truth[:, :4], pred[:, :4]))
        all_tp.append(tp.numpy())
        all_conf.extend(pred[:, 4].tolist())
        all_cls.extend(pred[:, 5].tolist())
        all_targets.extend(truth[:, 4].tolist())
    if not set(classes).issubset(set(all_targets)):
        raise ValueError('Every evaluated class needs ground-truth support')
    metrics = ap_per_class(np.concatenate(all_tp), np.array(all_conf), np.array(all_cls),
                           np.array(all_targets), names=dict(enumerate(NAMES)))
    ap, ids = metrics[5], metrics[6]
    return dict(AP50=float(ap[:, 0].mean()), AP50_95=float(ap.mean()), images=len(records),
                per_class={NAMES[c]: dict(AP50=float(ap[i, 0]), AP50_95=float(ap[i].mean()),
                                         objects=all_targets.count(c)) for i, c in enumerate(ids)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('predict')
    p.add_argument('source', type=Path, help='Image or directory of images')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu', choices=['cpu', 'mps'])
    p.add_argument('--config', type=Path, default=CONFIG)
    e = sub.add_parser('evaluate')
    e.add_argument('manifest', type=Path, nargs='?', default=EVALUATION)
    e.add_argument('--output', type=Path, required=True)
    e.add_argument('--device', default='cpu', choices=['cpu', 'mps'])
    e.add_argument('--predictions', type=Path, help='Reuse a frozen prediction receipt')
    e.add_argument('--config', type=Path, default=CONFIG)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.command == 'predict':
        paths = sorted(p for p in args.source.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}) if args.source.is_dir() else [args.source]
        if not paths:
            raise ValueError('No images found')
        records = [dict(name=str(p.resolve()), image_path=str(p.resolve()), sha256=sha(p)) for p in paths]
        save(args.output, predict(records, args.device, args.config))
    else:
        manifest = read(args.manifest)
        if manifest['names'] != NAMES:
            raise ValueError('Unexpected manifest class mapping')
        records, classes = manifest['records'], manifest['evaluated_classes']
        validate_annotations(records, classes)
        verify_images(records)
        receipt = read(args.predictions) if args.predictions else predict(records, args.device, args.config)
        receipt = validate_receipt(receipt, records, read(args.config), sha(args.config))
        result = score_predictions(records, receipt['predictions'], classes)
        save(args.output, dict(result, independent_test=False, manifest_sha256=sha(args.manifest),
                               annotation_status=manifest['annotation_status'], prediction_receipt=receipt))
        print(json.dumps({k: v for k, v in result.items()}, indent=2))


if __name__ == '__main__':
    main()
