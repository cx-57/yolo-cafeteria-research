"""Unified V5 trials. Specialist training is deliberately unavailable at this stage."""
from collections import Counter
import csv
import json
import math
from pathlib import Path
import platform
import time

import yaml

from v5_dataset import NAMES, sha, safe_path, verify_dataset, write_json

TRIALS = ('baseline', 'U1', 'U2', 'U3', 'U4', 'U5', 'U6')
SEEDS = (57, 58, 59)


def operating_point(confidence, precision, recall):
    # Zero predictions do not establish an attainable precision target.
    options = [(float(r), float(x)) for x, p, r in zip(confidence, precision, recall)
               if p >= .8 and r > 0]
    if not options:
        return dict(recall_at_precision_80=0.0, threshold_at_precision_80=None)
    r, x = max(options, key=lambda pair: (pair[0], -pair[1]))
    return dict(recall_at_precision_80=r, threshold_at_precision_80=x)


def weak_classes(record):
    return [name for name in NAMES if record['per_class'][name]['AP50'] < .8
            or record['per_class'][name]['recall_at_precision_80'] < .7]


def valid_completed(record, dataset_sha):
    if (record.get('status') != 'complete' or record.get('smoke') is not False
            or record.get('dataset_sha256') != dataset_sha
            or record.get('checkpoint_names') != NAMES
            or record.get('test_evaluated') is not False
            or record.get('trial') not in TRIALS or record.get('seed') not in SEEDS):
        return False
    if (type(record.get('completed_epochs')) is not int or type(record.get('planned_epochs')) is not int
            or record['completed_epochs'] < record['planned_epochs']
            or record['planned_epochs'] < (80 if record.get('recipe', {}).get('epochs') == 80 else 40)):
        return False
    per_class = record.get('per_class', {})
    if set(per_class) != set(NAMES):
        return False
    for values in per_class.values():
        for k in ('AP50', 'AP50_95', 'recall_at_precision_80'):
            value = values.get(k)
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                return False
        threshold = values.get('threshold_at_precision_80')
        if threshold is not None and (type(threshold) not in (float, int)
                                      or not math.isfinite(threshold) or not 0 <= threshold <= 1):
            return False
        if values['recall_at_precision_80'] > 0 and threshold is None:
            return False
    return True


def select_candidate(runs):
    by_trial = {r['trial']: r for r in runs if r['seed'] == 57}
    baseline = by_trial['baseline']
    weak = weak_classes(baseline)
    acceptable = [r for r in by_trial.values() if all(
        r['per_class'][n]['AP50_95'] >= baseline['per_class'][n]['AP50_95'] - .02 - 1e-12
        for n in NAMES)]
    # Preserve baseline on ties or when it has no weak classes.
    if not weak:
        return baseline
    return max(acceptable, key=lambda r: (
        sum(r['per_class'][n]['AP50_95'] for n in weak) / len(weak),
        r['trial'] == 'baseline', -TRIALS.index(r['trial'])))


def trial_recipe(trial, seed, completed=()):
    if trial not in TRIALS or seed not in SEEDS:
        raise ValueError('Unknown trial or seed; use baseline/U1-U6 and seeds 57/58/59')
    recipe = dict(pretrained='yolo11s.pt', epochs=40, patience=40, imgsz=800, batch=4,
                  optimizer='AdamW', lr0=.0005, lrf=.01, weight_decay=.0005,
                  cos_lr=True, seed=seed, deterministic=True, workers=0,
                  amp=True, cache=False, nbs=64, degrees=7, translate=.1, scale=.25,
                  fliplr=.5, flipud=0, mosaic=.5, mixup=0, close_mosaic=5,
                  plots=True, save=True, save_period=-1, balance=False)
    changes = dict(U1=dict(scale=.1, mosaic=.1, close_mosaic=10),
                   U2=dict(imgsz=1024, batch=2), U3=dict(balance=True),
                   U4=dict(lr0=.0002), U5=dict(pretrained='yolo11m.pt', batch=2))
    if trial == 'U6':
        by = {r['trial']: r for r in completed if r.get('seed') == 57}
        if not all(k in by for k in TRIALS[:-1]):
            raise ValueError('U6 requires completed baseline and U1 through U5 prerequisites')
        baseline = by['baseline']
        weak = weak_classes(baseline)
        supported = [r for k, r in by.items() if k in changes and
                     all(r['per_class'][n]['AP50_95'] >= baseline['per_class'][n]['AP50_95'] - .02 - 1e-12
                         for n in NAMES) and any(
                             r['per_class'][n]['AP50_95'] >= baseline['per_class'][n]['AP50_95'] + .02
                             or r['per_class'][n]['recall_at_precision_80'] >=
                             baseline['per_class'][n]['recall_at_precision_80'] + .05 for n in weak)]
        if supported:
            for r in sorted(supported, key=lambda r: TRIALS.index(r['trial'])):
                recipe.update(changes[r['trial']])
        else:
            recipe.update(epochs=80, patience=80)
    elif trial in changes:
        recipe.update(changes[trial])
    return recipe


def balanced_training_paths(records, weak):
    weak_ids = {NAMES.index(n) for n in weak}
    paths = []
    for r in records:
        if r['split'] != 'train':
            continue
        positive = any(int(c) in weak_ids and count > 0 for c, count in r['counts'].items())
        paths.extend([r['image_path']] * (3 if positive else 1))
    return paths


def verify_artifacts(root, receipt):
    root = Path(root).resolve()
    for rel, expected in receipt.get('artifacts', {}).items():
        safe_path(rel)
        path = root / rel
        if not path.is_file() or path.is_symlink() or sha(path) != expected:
            raise ValueError(f'Changed run artifact: {rel}')


def load_runs(root, dataset_sha):
    records = []
    root = Path(root)
    if not root.exists():
        return records
    for path in sorted(root.glob('*/run.json')):
        record = json.loads(path.read_text())
        if record.get('dataset_sha256') != dataset_sha:
            continue
        if record.get('status') == 'complete':
            required = {'fit/weights/best.pt', 'fit/results.csv', 'configuration.json', 'train.txt',
                        'run_data.yaml', 'validation_stats.json', 'curves.json'}
            if not required <= record.get('artifacts', {}).keys():
                raise ValueError(f'Run is missing required artifact provenance: {path}')
            verify_artifacts(path.parent, record)
            cfg = json.loads((path.parent / 'configuration.json').read_text())
            if any(record.get(k) != cfg.get(k) for k in ('trial', 'seed', 'smoke', 'dataset_sha256', 'recipe')):
                raise ValueError(f'Run configuration differs from receipt: {path}')
            if record.get('planned_epochs') != cfg['recipe']['epochs']:
                raise ValueError('Run epoch budget differs from configuration')
            with (path.parent / 'fit/results.csv').open() as f:
                actual_epochs = len(list(csv.DictReader(f)))
            if actual_epochs != record.get('completed_epochs'):
                raise ValueError('Training log epoch count differs from receipt')
        records.append(record)
    return records


def summarize_stage(runs, dataset_sha):
    completed = [r for r in runs if valid_completed(r, dataset_sha)]
    keys = [(r['trial'], r['seed']) for r in completed]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate completed trial/seed; use a separate experiment root')
    missing = [k for k in TRIALS if (k, 57) not in keys]
    report = dict(specialists_allowed=False, missing_trials=missing,
                  completed_runs=[dict(trial=k, seed=s) for k, s in keys],
                  status='unified_trials_required', test_evaluated=False)
    if missing:
        return report
    selected = select_candidate(completed)
    report['selected_unified_trial'] = selected['trial']
    report['baseline_weak_classes'] = weak_classes(next(r for r in completed if r['trial'] == 'baseline' and r['seed'] == 57))
    needed = [(k, seed) for k in dict.fromkeys(('baseline', selected['trial'])) for seed in (58, 59)
              if (k, seed) not in keys]
    report['required_confirmation_runs'] = [dict(trial=k, seed=s) for k, s in needed]
    report['status'] = 'confirmation_runs_required' if needed else 'uncertainty_analysis_required'
    report['note'] = ('No specialist eligibility has been established. After repeats, inspect group support, '
                      'paired group-bootstrap intervals, seed variability and remaining class deficits '
                      'before implementing a specialist. This status cannot be overridden by a CLI flag.')
    return report


def stage_report(dataset, runs):
    verify_dataset(dataset)
    dataset_sha = sha(Path(dataset) / 'dataset_receipt.json')
    return summarize_stage(load_runs(runs, dataset_sha), dataset_sha)


def checkpoint_names(model):
    names = model.names
    return [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)


def validate_model(model):
    if checkpoint_names(model) != NAMES:
        raise ValueError('Checkpoint must have exactly six retained classes, in the fixed order')


def verify_runtime(previous, current):
    if any(previous.get(k) != current.get(k) for k in ('python', 'torch', 'ultralytics')):
        raise ValueError('Cannot resume in a different Python/Torch/Ultralytics environment')


def resume_action(previous, checkpoint, epochs):
    if previous.get('phase') == 'evaluation' and previous.get('completed_epochs') == epochs:
        return 'evaluate'
    epoch = checkpoint.get('epoch')
    if type(epoch) is not int or not 0 <= epoch < epochs - 1 or checkpoint.get('optimizer') is None:
        raise ValueError('Checkpoint is not resumable; refusing to restart training')
    return 'train'


def clear_label_caches(dataset):
    # cache=False disables image caching only. Rebuild derived label caches from
    # the verified text labels, including before validation/recovery loads.
    for split in ('train', 'val', 'test'):
        path = Path(dataset) / 'labels' / (split + '.cache')
        if path.is_file() or path.is_symlink():
            path.unlink()


def evaluate(model, data_file, output, recipe, device, batch):
    """Capture per-image matching stats before Ultralytics clears them."""
    from ultralytics.models.yolo.detect.val import DetectionValidator

    class CaptureValidator(DetectionValidator):
        def update_metrics(self, preds, batch_data):
            if not hasattr(self, 'v5_image_names'):
                self.v5_image_names = []
            self.v5_image_names.extend(Path(p).name for p in batch_data['im_file'])
            super().update_metrics(preds, batch_data)

        def get_stats(self):
            per_image = []
            for i, name in enumerate(self.v5_image_names):
                per_image.append(dict(name=name, **{
                    k: self.metrics.stats[k][i].tolist() for k in ('tp', 'conf', 'pred_cls', 'target_cls')}))
            write_json(output / 'validation_stats.json', per_image)
            return super().get_stats()

    validate_model(model)
    clear_label_caches(Path(yaml.safe_load(Path(data_file).read_text())['path']))
    metrics = model.val(validator=CaptureValidator, data=str(data_file), split='val',
                        imgsz=recipe['imgsz'], batch=batch, device=device, workers=0,
                        conf=.001, iou=.7, max_det=300, augment=False,
                        plots=True, save_txt=True, save_conf=True,
                        project=str(output), name='validation', exist_ok=False)
    box = metrics.box
    if set(int(c) for c in box.ap_class_index) != set(range(6)):
        raise ValueError('Validation metrics lack retained-class support')
    per_class = {}
    for j, c in enumerate(box.ap_class_index):
        per_class[NAMES[int(c)]] = dict(
            AP50=float(box.ap50[j]), AP50_95=float(box.ap[j]),
            precision_at_mean_best_f1=float(box.p[j]), recall_at_mean_best_f1=float(box.r[j]),
            **operating_point(box.px, box.p_curve[j], box.r_curve[j]))
    curves = dict(confidence=box.px.tolist(), class_indices=box.ap_class_index.tolist(),
                  precision=box.p_curve.tolist(), recall=box.r_curve.tolist())
    write_json(output / 'curves.json', curves)
    return per_class, {k: float(v) for k, v in metrics.speed.items()}


def run_trial(dataset, output, trial, seed=57, device='mps', batch=None, smoke=False, resume=False):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    data = verify_dataset(dataset)
    dataset_sha = sha(dataset / 'dataset_receipt.json')
    runs = load_runs(output.parent, dataset_sha)
    completed = [r for r in runs if valid_completed(r, dataset_sha)]
    baseline = next((r for r in completed if r['trial'] == 'baseline' and r['seed'] == 57), None)
    if not smoke and trial != 'baseline' and baseline is None:
        raise ValueError('Complete the baseline at seed 57 before variants')
    if not smoke and seed != 57:
        report = summarize_stage(completed, dataset_sha)
        if report['missing_trials'] or trial not in ('baseline', report['selected_unified_trial']):
            raise ValueError('Complete all unified variants before baseline/selected-candidate confirmations')
    recipe = trial_recipe(trial, seed, completed)
    if batch is not None:
        if type(batch) is not int or batch < 1:
            raise ValueError('Batch must be a positive integer')
        recipe['batch'] = batch
    if smoke:
        recipe.update(epochs=2, patience=2, close_mosaic=0)
    config = dict(schema=1, trial=trial, seed=seed, dataset_sha256=dataset_sha,
                  recipe=recipe, smoke=smoke, device=device)
    if output.exists() and not resume:
        raise FileExistsError(f'Refusing to overwrite trial: {output}')
    from ultralytics import YOLO
    import ultralytics
    import torch
    runtime = dict(python=platform.python_version(), torch=torch.__version__,
                   ultralytics=ultralytics.__version__, platform=platform.platform())
    action = 'train'
    if device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS unavailable; choose --device cpu or a CUDA device explicitly')
    if resume:
        previous = json.loads((output / 'run.json').read_text())
        if json.loads((output / 'configuration.json').read_text()) != config or previous['status'] != 'failed':
            raise ValueError('Resume requires a failed run with exactly matching data/configuration')
        verify_runtime(previous.get('runtime', {}), runtime)
        verify_artifacts(output, {'artifacts': previous.get('input_artifact_hashes', {})})
        if set(previous.get('input_artifact_hashes', {})) != {'configuration.json', 'run_data.yaml', 'train.txt'}:
            raise ValueError('Failed run is missing input provenance for safe resume')
        if previous.get('phase') == 'evaluation' and previous.get('completed_epochs') == recipe['epochs']:
            action = 'evaluate'
            required = {'fit/weights/best.pt', 'fit/results.csv'}
            if set(previous.get('training_artifact_hashes', {})) != required:
                raise ValueError('Missing completed-training provenance for evaluation recovery')
            verify_artifacts(output, {'artifacts': previous['training_artifact_hashes']})
            model = YOLO(str(output / 'fit/weights/best.pt'))
        else:
            last = output / 'fit/weights/last.pt'
            if not last.is_file() or previous.get('last_checkpoint_sha256') != sha(last):
                raise ValueError('No verified resumable last.pt from the failed run')
            model = YOLO(str(last))
            action = resume_action(previous, model.ckpt, recipe['epochs'])
            args = model.ckpt.get('train_args', {})
            if any(args.get(k) != value for k, value in
                   dict(data=str(output / 'run_data.yaml'), epochs=recipe['epochs'], seed=seed).items()):
                raise ValueError('Checkpoint training arguments disagree with recorded run')
        validate_model(model)
    else:
        output.mkdir(parents=True)
        write_json(output / 'configuration.json', config)
        train_paths = balanced_training_paths(data['images'], weak_classes(baseline)) if recipe['balance'] and baseline else [
            r['image_path'] for r in data['images'] if r['split'] == 'train']
        (output / 'train.txt').write_text(''.join(str(dataset / p) + '\n' for p in train_paths))
        (output / 'run_data.yaml').write_text(yaml.safe_dump(dict(
            path=str(dataset), train=str(output / 'train.txt'), val=str(dataset / 'val.txt'), nc=6, names=NAMES), sort_keys=False))
        model = YOLO(recipe['pretrained'])
    record = dict(config, status='running', phase='training', planned_epochs=recipe['epochs'],
                  test_evaluated=False, annotation_status=data['annotation_status'],
                  runtime=runtime)
    record['input_artifact_hashes'] = {p: sha(output / p) for p in
                                      ('configuration.json', 'run_data.yaml', 'train.txt')}
    pretrained = Path(getattr(model, 'ckpt_path', '') or recipe['pretrained'])
    if pretrained.is_file():
        record['initial_checkpoint_sha256'] = sha(pretrained)
    started = time.monotonic()
    try:
        if resume:
            # Keep the old discoverable failure receipt until all checkpoint
            # loading and provenance validation above have succeeded.
            write_json(output / f'failure-{time.time_ns()}.json', previous)
            (output / 'run.json').unlink()
        clear_label_caches(dataset)
        train_args = {k: v for k, v in recipe.items() if k not in ('pretrained', 'balance')}
        if action == 'evaluate':
            record['completed_epochs'] = previous['completed_epochs']
            record['phase'] = 'evaluation'
            record['training_artifact_hashes'] = previous['training_artifact_hashes']
            partials = [output / p for p in ('validation', 'validation_stats.json', 'curves.json')
                        if (output / p).exists()]
            if partials:
                archive = output / f'failed-validation-{time.time_ns()}'
                archive.mkdir()
                for path in partials:
                    path.rename(archive / path.name)
        elif resume:
            model.train(resume=True, device=device)
        else:
            model.train(data=str(output / 'run_data.yaml'), device=device,
                        project=str(output), name='fit', exist_ok=False, **train_args)
        if action != 'evaluate':
            record['completed_epochs'] = model.trainer.epoch + 1
        if record['completed_epochs'] != recipe['epochs']:
            raise ValueError('Training stopped before completing the declared budget')
        best = output / 'fit/weights/best.pt'
        record['training_artifact_hashes'] = {p: sha(output / p) for p in
                                             ('fit/weights/best.pt', 'fit/results.csv')}
        record['phase'] = 'evaluation'
        trained = YOLO(str(best))
        validate_model(trained)
        # Revalidate all input hashes after training before accepting metrics.
        verify_dataset(dataset)
        verify_artifacts(output, {'artifacts': record['input_artifact_hashes']})
        if sha(dataset / 'dataset_receipt.json') != dataset_sha:
            raise ValueError('Dataset receipt changed during training')
        record['per_class'], record['validation_speed_ms'] = evaluate(
            trained, output / 'run_data.yaml', output, recipe, device, recipe['batch'])
        stats = json.loads((output / 'validation_stats.json').read_text())
        expected_images = {Path(r['image_path']).name for r in data['images'] if r['split'] == 'val'}
        if len(stats) != len(expected_images) or {r['name'] for r in stats} != expected_images:
            raise ValueError('Captured validation cohort differs from the fixed split')
        record.update(status='complete', checkpoint_names=checkpoint_names(trained),
                      checkpoint_sha256=sha(best), wall_seconds=time.monotonic()-started)
        artifacts = ['configuration.json', 'run_data.yaml', 'train.txt', 'fit/results.csv',
                     'fit/weights/best.pt', 'validation_stats.json', 'curves.json']
        record['artifacts'] = {p: sha(output / p) for p in artifacts}
        write_json(output / 'run.json', record)
    except BaseException as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                      wall_seconds=time.monotonic()-started)
        last = output / 'fit/weights/last.pt'
        if last.is_file():
            record['last_checkpoint_sha256'] = sha(last)
        if not (output / 'run.json').exists():
            write_json(output / 'run.json', record)
        raise
    return record
