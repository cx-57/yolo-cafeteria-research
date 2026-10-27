"""Isolated packet training with a group-separated internal monitor and fixed final evaluation."""
import argparse
import gc
import json
import os
import random
import datetime
import re
from pathlib import Path

import packet_specialist as ps
import specialists as s

from storage import ROOT, DATA, EVALUATION
from storage import RUNTIME as CACHE
RUN = CACHE / 'packet-big-run-20261027'
REPORT = ROOT / 'runs/packet-big-run-20261027'


def grouped_monitor_split(records, seed=42, fraction=.2):
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, record in enumerate(records):
        keys = [(field, record.get(field)) for field in ('group', 'sha256', 'pixels')]
        keys += [(field, record.get('provenance', {}).get(field)) for field in ('source_sha1', 'pageid')]
        for field, value in keys:
            if not value:
                continue
            key = field, str(value)
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    components = {}
    for i, record in enumerate(records):
        components.setdefault(find(i), []).append(record)
    groups = sorted(components.values(), key=lambda g: min(r['name'] for r in g))
    if len(groups) < 5:
        raise ValueError('Insufficient independent groups for internal monitor')
    random.Random(seed).shuffle(groups)
    n = max(1, round(len(groups) * fraction))
    monitor = [r for group in groups[:n] for r in group]
    train = [r for group in groups[n:] for r in group]
    if not all(any(r['boxes'] for r in rows) for rows in (train, monitor)):
        raise ValueError('Both training and monitor require target support')
    return train, monitor


def verify_prepared_files(run, source, fitting, monitor):
    import yaml
    expected = dict(path=str(run / 'data'), train='images/train', val='images/val', names=['condiment_packet'])
    if yaml.safe_load((run / 'data/data.yaml').read_text()) != expected:
        raise ValueError('Prepared training YAML changed')
    # Validate full transitive source/identity separation with the existing packet validator.
    ps.validate_records([dict(r, split=split) for split, rows in [('train', fitting), ('val', monitor)] for r in rows])
    for split, rows in [('train', fitting), ('val', monitor)]:
        for kind in ('images', 'labels'):
            expected_names = {r['name'] if kind == 'images' else str(Path(r['name']).with_suffix('.txt')) for r in rows}
            actual = {str(p.relative_to(run / 'data' / kind / split)) for p in (run / 'data' / kind / split).rglob('*') if p.is_file()}
            if actual != expected_names:
                raise ValueError('Prepared dataset membership changed')
        for r in rows:
            image = run / 'data/images' / split / r['name']
            expected_image = source / 'images/train' / r['name']
            if not image.is_symlink() or image.resolve() != expected_image.resolve() or s.sha(image) != r['sha256']:
                raise ValueError('Prepared image link/content changed')
            label = run / 'data/labels' / split / Path(r['name']).with_suffix('.txt')
            if label.read_text() != ''.join(ps.yolo_row(box) + '\n' for box in r['boxes']):
                raise ValueError('Prepared annotation changed')


def prepare(epochs=40):
    if (RUN / 'split.json').exists():
        raise FileExistsError('Run already prepared; use --run to execute the frozen prepared run')
    import yaml
    source = CACHE / 'packet-broad-v16'
    ps.RUNTIME = source
    ps.REVIEWS = DATA / 'packet-focus/broad-v16/reviews.json'
    original = ps.verify()
    shared_path = EVALUATION
    shared = s.read(shared_path)
    blocked = {r['sha256'] for r in shared['records']}
    raw_pixels = {r.get('pixels_raw') for r in shared['records']} - {None}
    shared_groups = {r.get('group') for r in shared['records']} - {None, ''}
    records = [r for r in original['records'] if r['split'] == 'train' and r['sha256'] not in blocked
               and r['pixels'] not in raw_pixels and r['group'] not in shared_groups]
    fitting, monitor = grouped_monitor_split(records)
    components = {'train': fitting, 'val': monitor}
    for split, rows in components.items():
        for kind in ('images', 'labels'):
            (RUN / 'data' / kind / split).mkdir(parents=True, exist_ok=True)
        for record in rows:
            image = source / 'images/train' / record['name']
            if s.sha(image) != record['sha256']:
                raise ValueError('Training source changed')
            (RUN / 'data/images' / split / record['name']).symlink_to(image)
            (RUN / 'data/labels' / split / Path(record['name']).with_suffix('.txt')).write_text(
                ''.join(ps.yolo_row(box) + '\n' for box in record['boxes']))
    (RUN / 'data/data.yaml').write_text(yaml.safe_dump(dict(path=str(RUN / 'data'), train='images/train', val='images/val', names=['condiment_packet'])))
    initial = ROOT / 'weights/yolo11s.pt'
    config = ROOT / 'configs/specialists.json'
    recipe = dict(source_manifest_sha256=s.sha(source / 'manifest.json'), reviews_sha256=s.sha(ps.REVIEWS),
                  shared_manifest_sha256=s.sha(shared_path), baseline_config_sha256=s.sha(config),
                  pretrained=str(initial), pretrained_sha256=s.sha(initial), epochs=epochs, patience=8,
                  batch=1, imgsz=640, seed=42, lr0=.0005, nbs=16, freeze=10, mosaic=.25,
                  selection='highest internal-monitor AP50-95, never shared development',
                  independent_test=False, automatic_promotion=False,
                  counts={split: dict(images=len(rows), boxes=sum(len(r['boxes']) for r in rows)) for split, rows in components.items()})
    ps.save(RUN / 'split.json', dict(recipe=recipe, fitting=fitting, monitor=monitor))
    ps.save(REPORT / 'recipe.json', recipe)
    print(json.dumps(recipe, indent=2), flush=True)


def run():
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    torch.set_num_threads(2)
    if (RUN / 'train').exists():
        raise FileExistsError('Training already started; refusing to overwrite or silently restart')
    split = s.read(RUN / 'split.json')
    recipe = split['recipe']
    initial = Path(recipe['pretrained'])
    if s.sha(initial) != recipe['pretrained_sha256']:
        raise ValueError('Pretrained source changed')
    if s.sha(ROOT / 'configs/specialists.json') != recipe['baseline_config_sha256']:
        raise ValueError('Active pipeline changed since preparation')
    if s.sha(CACHE / 'packet-broad-v16/manifest.json') != recipe['source_manifest_sha256'] or s.sha(DATA / 'packet-focus/broad-v16/reviews.json') != recipe['reviews_sha256']:
        raise ValueError('Source manifest or review ledger changed')
    if s.sha(EVALUATION) != recipe['shared_manifest_sha256']:
        raise ValueError('Shared development manifest changed')
    verify_prepared_files(RUN, CACHE / 'packet-broad-v16', split['fitting'], split['monitor'])
    # Recheck both source identities and the absence of cross-split aliases.
    for rows in (split['fitting'], split['monitor']):
        for r in rows:
            p = CACHE / 'packet-broad-v16/images/train' / r['name']
            if s.sha(p) != r['sha256']:
                raise ValueError('Training image changed after preparation')

    class Trainer(DetectionTrainer):
        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
            return super().get_dataloader(dataset_path, 1, rank, mode)
        def _model_train(self):
            super()._model_train()
            for module in self.model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
        def final_eval(self):
            pass

    def progress(trainer):
        ps.save(REPORT / 'status.json', dict(state='training', pid=os.getpid(), epoch=trainer.epoch + 1,
                epoch_cap=recipe['epochs'], monitor_metrics={k: float(v) for k, v in trainer.metrics.items()},
                best_fitness=float(trainer.best_fitness), current_pipeline_unchanged=True))
    def check_bn(trainer):
        if any(m.training for m in trainer.model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)):
            raise RuntimeError('Tiny-batch BatchNorm must remain fixed')

    ps.save(REPORT / 'status.json', dict(state='starting', pid=os.getpid(), current_pipeline_unchanged=True))
    model = YOLO(str(initial))
    if model.model.model[-1].nc != 80:
        raise ValueError('Expected fresh COCO pretrained detector')
    model.add_callback('on_fit_epoch_end', progress)
    model.add_callback('on_train_batch_start', check_bn)
    model.train(trainer=Trainer, data=str(RUN / 'data/data.yaml'), epochs=recipe['epochs'], patience=8,
                batch=1, imgsz=640, device='mps', workers=0, optimizer='AdamW', lr0=.0005,
                lrf=.1, nbs=16, freeze=10, seed=42, cache=False, warmup_epochs=2,
                warmup_bias_lr=0, mosaic=.25, close_mosaic=5, mixup=0, fliplr=.5,
                conf=.001, iou=.7, max_det=300, plots=False, save=True, save_period=-1,
                project=str(RUN), name='train', exist_ok=False)
    del model
    gc.collect()
    torch.mps.empty_cache()
    best = RUN / 'train/weights/best.pt'
    candidate = s.read(ROOT / 'configs/specialists.json')
    for spec in candidate['models']:
        if spec['id'] == 'packet':
            spec.update(weights=str(best), sha256=s.sha(best), historical_receipt=str(REPORT / 'recipe.json'))
    candidate_path = REPORT / 'candidate-config.json'
    ps.save(candidate_path, candidate)
    shared_path = EVALUATION
    if s.sha(shared_path) != recipe['shared_manifest_sha256']:
        raise ValueError('Shared development cohort changed')
    shared = s.read(shared_path)
    s.validate_annotations(shared['records'], list(range(8)))
    receipt = s.predict(shared['records'], 'mps', candidate_path)
    metrics = s.score_predictions(shared['records'], receipt['predictions'], list(range(8)))
    baseline = s.read(ROOT / 'results/baseline.json')
    ps.save(REPORT / 'shared-development.json', dict(metrics, prediction_receipt=receipt,
             baseline_AP50=baseline['AP50'], baseline_AP50_95=baseline['AP50_95'],
             independent_test=False, promoted=False, manifest_sha256=s.sha(shared_path)))
    if s.sha(ROOT / 'configs/specialists.json') != recipe['baseline_config_sha256']:
        raise ValueError('Active configuration changed during candidate run')
    ps.save(REPORT / 'status.json', dict(state='complete', AP50=metrics['AP50'], AP50_95=metrics['AP50_95'],
             packet=metrics['per_class']['condiment_packet'], promoted=False, current_pipeline_unchanged=True))
    print('COMPLETE', json.dumps(metrics), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Execute an already prepared run selected with --name')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--epochs', type=int, choices=range(1, 81), default=40, metavar='1-80')
    parser.add_argument('--name', default='packet-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    args = parser.parse_args()
    if not re.fullmatch(r'packet-[A-Za-z0-9._-]+', args.name):
        parser.error('--name must start with packet- and contain only letters, digits, dots, underscores or hyphens')
    RUN = CACHE / args.name
    REPORT = ROOT / 'runs' / args.name
    try:
        if not args.run:
            prepare(args.epochs)
        if not args.prepare_only:
            run()
    except Exception as error:
        ps.save(REPORT / 'failure.json', dict(error=repr(error), pid=os.getpid(), current_pipeline_unchanged=True))
        raise
