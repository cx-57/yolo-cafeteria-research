"""Build, verify, fine-tune and evaluate the isolated plate/wrapper YOLO11n experiment."""
import argparse
import copy
import csv
import json
import math
from pathlib import Path
import shutil

import yaml

from experiment import check_output, label_path
from focus_three_class import (apply_manual_reviews, audit, digest, fingerprint,
                               continuation_checkpoint,
                               minimum_class_callback, pixels, scene_reviews,
                               validate_image_file, validate_resume_epoch,
                               verify as verify_source)
from prepare import IMAGE_SUFFIXES, PROJECT_ROOT

NAMES = ['plate', 'wrapper']
SOURCE = PROJECT_ROOT / 'data/focus-three-class-improvement/capture-grouped-clean'
DATA = PROJECT_ROOT / 'data/plate-wrapper-focus/reviewed-v2'
MODEL = PROJECT_ROOT / 'model/plate-wrapper-focus/two-class90-reviewed'
PREVIOUS_MODEL = PROJECT_ROOT / 'model/plate-wrapper-focus/two-class90'
BASELINE = PROJECT_ROOT / 'model/plate-wrapper-focus/target.json'
REVIEWS = PROJECT_ROOT / 'data/plate-wrapper-focus/training-reviews.json'
CORRECTIONS = PROJECT_ROOT / 'data/plate-wrapper-focus/label-corrections-v2.json'
CLOSED = {'corrected', 'focus_verified', 'full_verified', 'corrected_and_signed_off',
          'verified_unchanged', 'external_signed_off', 'corrected_three_class_only',
          'previously_signed_off'}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def project_records(records, review_files, allow_validation_relabel=False):
    """A negative requires a completed three-class review, including bottle-only images."""
    frozen = {r['name']: r['rows'] for r in records if r.get('split') == 'val'}
    records = copy.deepcopy(records)
    apply_manual_reviews(records, review_files)
    for record in records:
        if (not allow_validation_relabel and record['name'] in frozen and
                record['rows'] != frozen[record['name']]):
            raise ValueError('Validation labels are frozen: ' + record['name'])
        if record['review_status'] not in CLOSED or record.get('accepted') is False:
            raise ValueError('Unresolved review: ' + record['name'])
        if not record['rows'] and not record['verified_absence']:
            raise ValueError('Unverified source negative: ' + record['name'])
        record['rows'] = [row for row in record['rows'] if row.split()[0] in ('0', '1')]
        record['verified_absence'] = not record['rows']
    return records


def validation_changes(original, records):
    prior = {r['name']: [row for row in r['rows'] if row.split()[0] in ('0', '1')]
             for r in original if r['split'] == 'val'}
    return [dict(name=r['name'], before=prior[r['name']], after=r['rows']) for r in records
            if r['split'] == 'val' and r['rows'] != prior[r['name']]]


def build(source, data, review_files=(), rebaseline=False):
    source, data = source.resolve(), data.resolve()
    baseline = verify_source(source)
    check_output(data, [source, PROJECT_ROOT / 'data/training-v2'])
    records = project_records(baseline['records'], review_files, allow_validation_relabel=rebaseline)
    counts = audit(records, baseline['selected'], names=NAMES)
    manifest = dict(selected=len(records), records=records, counts=counts,
                    source=str(source), source_fingerprint=baseline['fingerprint'],
                    source_validation_protocol=baseline.get('validation_protocol'),
                    scene_reviews=baseline.get('scene_reviews', []),
                    manual_reviews=[dict(path=str(p.resolve()), sha256=digest(p)) for p in review_files],
                    benchmark_revised=rebaseline,
                    validation_label_changes=validation_changes(baseline['records'], records),
                    fingerprint=fingerprint(records),
                    validation_protocol=fingerprint([r for r in records if r['split'] == 'val']),
                    negative_policy='Completed three-class review verifies absence of both remaining classes.')
    print(json.dumps(counts, indent=2), flush=True)
    for record in records:
        for kind in ('images', 'labels'):
            destination = data / kind / record['split'] / record['name']
            if kind == 'labels':
                destination = destination.with_suffix('.txt')
            destination.parent.mkdir(parents=True, exist_ok=True)
            if kind == 'images':
                # Independent files: training must never rewrite a source through a hard link.
                shutil.copy2(source / kind / record['split'] / record['name'], destination)
            else:
                destination.write_text('\n'.join(record['rows']) + ('\n' if record['rows'] else ''))
    save_json(data / 'manifest.json', manifest)
    (data / 'data.yaml').write_text(yaml.safe_dump(dict(path=str(data), train='images/train',
                                                      val='images/val', names=NAMES), sort_keys=False))
    return verify(data)


def verify(data):
    data = data.resolve()
    if (data / 'quality_hold.json').exists():
        raise ValueError('Dataset quality hold: ' + (data / 'quality_hold.json').read_text())
    manifest = json.loads((data / 'manifest.json').read_text())
    source = verify_source(Path(manifest['source']))
    if (source['fingerprint'] != manifest['source_fingerprint'] or
            source.get('validation_protocol') != manifest['source_validation_protocol']):
        raise ValueError('Source provenance changed')
    reviews = []
    for evidence in manifest['manual_reviews']:
        path = Path(evidence['path'])
        if digest(path) != evidence['sha256']:
            raise ValueError('Manual review evidence changed')
        reviews.append(path)
    expected = project_records(source['records'], reviews,
                               allow_validation_relabel=manifest.get('benchmark_revised', False))
    if (validation_changes(source['records'], expected) != manifest.get('validation_label_changes', [])):
        raise ValueError('Validation correction evidence changed')
    if expected != manifest['records'] or len(expected) != manifest['selected']:
        raise ValueError('Two-class projection no longer preserves source records')
    scene_reviews(expected, manifest['scene_reviews'])
    if manifest['scene_reviews'] != source.get('scene_reviews', []):
        raise ValueError('Source scene evidence changed')
    spec = dict(path=str(data), train='images/train', val='images/val', names=NAMES)
    if yaml.safe_load((data / 'data.yaml').read_text()) != spec:
        raise ValueError('Dataset YAML changed')
    for split in ('train', 'val'):
        names = {r['name'] for r in expected if r['split'] == split}
        actual = {p.name for p in (data / 'images' / split).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
        labels = {p.name for p in (data / 'labels' / split).glob('*.txt')}
        if actual != names or labels != {str(Path(n).with_suffix('.txt')) for n in names}:
            raise ValueError('Image/label membership changed')
    for record in expected:
        image = data / 'images' / record['split'] / record['name']
        validate_image_file(image)
        if (digest(image) != record['sha'] or pixels(image) != record['pixels'] or
                label_path(image).read_text().splitlines() != record['rows']):
            raise ValueError('Dataset bytes or labels changed: ' + record['name'])
    counts = audit(expected, manifest['selected'], names=NAMES)
    protocol = fingerprint([r for r in expected if r['split'] == 'val'])
    if (counts != manifest['counts'] or fingerprint(expected) != manifest['fingerprint'] or
            protocol != manifest['validation_protocol']):
        raise ValueError('Dataset fingerprint or counts changed')
    print('Two-class integrity passed: all source images/splits retained; boxes validated; '
          'negatives reviewed; no recorded hash/original/capture/scene overlap. '
          'Unknown related captures may remain.', flush=True)
    print(json.dumps(counts, indent=2), flush=True)
    return manifest


def metrics_result(metrics, manifest, filtered_bottle=False):
    box = metrics.box
    indices = list(map(int, box.ap_class_index))
    counts = [manifest['counts']['val']['objects'][name] for name in NAMES]
    support = counts + ([0] if filtered_bottle else [])
    if set(indices) != {0, 1} or len(indices) != 2 or list(map(int, metrics.nt_per_class)) != support:
        raise ValueError('Validation support differs from audited two-class data')
    per_class = {NAMES[c]: dict(precision=float(box.p[j]), recall=float(box.r[j]),
                               AP50=float(box.ap50[j]), AP50_95=float(box.ap[j]), objects=counts[c])
                 for j, c in enumerate(indices)}
    overall = dict(precision=float(box.mp), recall=float(box.mr), AP50=float(box.map50), AP50_95=float(box.map))
    if any(not math.isfinite(v) or not 0 <= v <= 1 for row in [overall, *per_class.values()]
           for key, v in row.items() if key != 'objects'):
        raise ValueError('Invalid evaluation metrics')
    return dict(per_class=per_class, overall=overall)


def comparison_path(manifest):
    return MODEL.parent / ('rebaseline-' + manifest['fingerprint'][:12]) / 'comparison.json'


def validate_warm_split(earlier, revised):
    def membership(records):
        return {r['name']: (r['split'], r['sha']) for r in records}
    if membership(earlier) != membership(revised):
        raise ValueError('Warm-start image membership/split changed across label revisions')


def validate_baseline_report(report, manifest):
    if (report['dataset_fingerprint'] != manifest['fingerprint'] or
            report['validation_protocol'] != manifest['validation_protocol'] or
            set(report['candidates']) != {'filtered_three_class', 'two_class_epoch1'}):
        raise ValueError('Corrected baseline report does not match this benchmark')
    for entry in report['candidates'].values():
        if digest(Path(entry['weights'])) != entry['weights_sha256']:
            raise ValueError('Baseline checkpoint weights changed')
        if {n: entry['metrics']['per_class'][n]['objects'] for n in NAMES} != manifest['counts']['val']['objects']:
            raise ValueError('Baseline object support changed')


def baseline(data):
    """Re-evaluate both preserved models on the SAME explicitly corrected benchmark."""
    manifest = verify(data)
    location = comparison_path(manifest)
    if location.exists():
        report = json.loads(location.read_text())
        validate_baseline_report(report, manifest)
        print(json.dumps(report, indent=2), flush=True)
        return report
    original = json.loads(BASELINE.read_text())
    selected = json.loads((PREVIOUS_MODEL / 'minimum_class_selection.json').read_text())
    previous = json.loads((PREVIOUS_MODEL / 'experiment.json').read_text())
    earlier_data = Path(previous['training_args']['data']).parent
    earlier = json.loads((earlier_data / 'manifest.json').read_text())
    if fingerprint(earlier['records']) != previous['dataset_fingerprint']:
        raise ValueError('Warm-start dataset evidence changed')
    validate_warm_split(earlier['records'], manifest['records'])
    if (original['dataset_fingerprint'] != manifest['source_fingerprint'] or
            original['validation_protocol'] != manifest['source_validation_protocol'] or
            previous['source_weights_sha256'] != original['weights_sha256']):
        raise ValueError('Initial checkpoint lineage does not match source capture assignments')
    from ultralytics import YOLO
    candidates = {}
    for name, path, expected_hash, names in [
        ('filtered_three_class', Path(original['weights']), original['weights_sha256'], NAMES + ['bottle']),
        ('two_class_epoch1', Path(selected['weights']), selected['weights_sha256'], NAMES)]:
        if digest(path) != expected_hash:
            raise ValueError('Baseline weights changed')
        model = YOLO(str(path))
        if list(model.names.values()) != names or model.model.yaml.get('scale') != 'n':
            raise ValueError('Expected the preserved YOLO11n baseline')
        filtered = len(names) == 3
        metrics = model.val(data=str(data / 'data.yaml'), split='val', device='mps', batch=4,
                            imgsz=640, workers=0, classes=[0, 1] if filtered else None,
                            augment=False, project=str(location.parent), name=name, plots=False)
        candidates[name] = dict(weights=str(path), weights_sha256=digest(path),
                                metrics=metrics_result(metrics, manifest, filtered_bottle=filtered))
    verify(data)
    report = dict(dataset_fingerprint=manifest['fingerprint'], validation_protocol=manifest['validation_protocol'],
                  candidates=candidates, validation_label_changes=manifest['validation_label_changes'],
                  interpretation='Both models re-evaluated on identical corrected labels; no training performed here.')
    validate_baseline_report(report, manifest)
    save_json(location, report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def evaluate(data, output, weights, target=.9):
    manifest = verify(data)
    experiment = json.loads((output / 'experiment.json').read_text())
    if (experiment['dataset_fingerprint'] != manifest['fingerprint'] or
            experiment['validation_protocol'] != manifest['validation_protocol']):
        raise ValueError('Checkpoint dataset provenance mismatch')
    if weights.resolve() not in [p.resolve() for p in (output / 'train/weights').glob('*.pt')]:
        raise ValueError('Evaluate a checkpoint from this experiment')
    weight_hash = digest(weights)
    location = output / ('evaluation-' + weight_hash[:12])
    if location.exists():
        saved = json.loads((location / 'metrics.json').read_text())
        if (saved['weights_sha256'] != weight_hash or
                saved['dataset_fingerprint'] != manifest['fingerprint'] or
                saved['validation_protocol'] != manifest['validation_protocol']):
            raise ValueError('Existing evaluation provenance mismatch')
        print(json.dumps(saved, indent=2), flush=True)
        return saved
    from ultralytics import YOLO
    model = YOLO(str(weights))
    if list(model.names.values()) != NAMES or model.model.model[-1].nc != 2:
        raise ValueError('Expected a real two-output model')
    metrics = model.val(data=str(data / 'data.yaml'), device='mps', imgsz=640, batch=4,
                        workers=0, split='val', classes=None, augment=False,
                        project=str(output), name=location.name, plots=True)
    result = metrics_result(metrics, manifest)
    result.update(weights=str(weights), weights_sha256=digest(weights),
                  dataset_fingerprint=manifest['fingerprint'], validation_protocol=manifest['validation_protocol'],
                  validation_images=manifest['counts']['val']['images'], target_AP50=target,
                  all_classes_meet_target=all(row['AP50'] >= target for row in result['per_class'].values()),
                  interpretation='Development validation reused during selection; independent cafeteria test still needed.')
    baseline = experiment['baseline_metrics']
    result['delta_AP50_from_initial_model'] = {
        name: result['per_class'][name]['AP50'] - baseline['per_class'][name]['AP50'] for name in NAMES}
    verify(data)
    save_json(location / 'metrics.json', result)
    save_json(output / 'evaluation.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def scaled_continuation(checkpoint, output, data, factor):
    """Scale the scheduler and optimizer rates together without resetting learning."""
    if not math.isfinite(factor) or not 0 < factor <= 1:
        raise ValueError('Learning-rate scale must be in (0, 1]')
    clone = continuation_checkpoint(checkpoint, output, data, 80)
    clone['optimizer'] = copy.deepcopy(checkpoint['optimizer'])
    clone['train_args']['lr0'] *= factor
    for group in clone['optimizer']['param_groups']:
        for key in ('lr', 'initial_lr'):
            if not math.isfinite(group[key]) or group[key] <= 0:
                raise ValueError('Invalid optimizer learning rate')
            group[key] *= factor
    return clone


def prepare_lr_branch(args, manifest):
    import torch
    source = args.continue_from.resolve()
    parent = source.parent.parent.parent
    if args.model.exists() or source.parent != parent / 'train/weights':
        raise ValueError('Choose a new model directory and a checkpoint inside experiment/train/weights')
    origin = json.loads((parent / 'experiment.json').read_text())
    if (origin['dataset_fingerprint'] != manifest['fingerprint'] or
            origin['validation_protocol'] != manifest['validation_protocol']):
        raise ValueError('Learning-rate comparison must preserve the audited dataset and validation')
    selection = json.loads((parent / 'minimum_class_selection.json').read_text())
    if source != Path(selection['weights']).resolve() or digest(source) != selection['weights_sha256']:
        raise ValueError('Branch from the verified selected checkpoint')
    measured = evaluate(args.data, parent, source, args.target)
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    clone = scaled_continuation(checkpoint, args.model, args.data, args.lr_scale)
    if Path(clone['train_args']['data']).resolve() != args.data / 'data.yaml':
        raise ValueError('Branch checkpoint uses different training data')
    weights = args.model / 'train/weights'
    weights.mkdir(parents=True)
    torch.save(clone, weights / 'last_resume.pt')
    copied = weights / 'best_min_ap50.pt'
    shutil.copy2(source, copied)
    selection.update(weights=str(copied), weights_sha256=digest(copied))
    save_json(args.model / 'minimum_class_selection.json', selection)
    with (parent / 'train/results.csv').open() as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        history = [r for r in reader if int(r['epoch']) <= checkpoint['epoch'] + 1]
    with (args.model / 'train/results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    validate_resume_epoch(weights / 'last_resume.pt', clone)
    save_json(args.model / 'experiment.json', {
        **origin, 'training_args': clone['train_args'], 'baseline_metrics': measured,
        'source_weights': str(source), 'source_weights_sha256': digest(source),
        'source_completed_epochs': checkpoint['epoch'] + 1, 'lr_scale': args.lr_scale,
        'initialization': 'Optimizer/EMA retained; only learning rates scaled. Prior CSV rows are inherited history.'})
    args.resume = True


def train(args):
    manifest = verify(args.data)
    from ultralytics import YOLO
    import torch
    if not torch.backends.mps.is_available():
        raise ValueError('MPS unavailable')
    output = args.model.resolve()
    if output.parent != MODEL.parent or output == MODEL.parent:
        raise ValueError('Use a new run directly under model/plate-wrapper-focus')
    if args.continue_from:
        if args.resume:
            raise ValueError('Use --continue-from only for a new branch, then --resume')
        prepare_lr_branch(args, manifest)
    baseline = json.loads(BASELINE.read_text())
    if (baseline['dataset_fingerprint'] != manifest['source_fingerprint'] or
            baseline['validation_protocol'] != manifest['source_validation_protocol']):
        raise ValueError('Baseline did not use this capture-grouped source split')
    initial = Path(baseline['weights'])
    if digest(initial) != baseline['weights_sha256']:
        raise ValueError('Baseline checkpoint changed')
    initial_metrics = baseline['metrics']
    if manifest.get('benchmark_revised'):
        report_path = comparison_path(manifest)
        if not report_path.is_file():
            raise ValueError('Corrected labels require rebaselining: run plate_wrapper.py baseline first')
        report = json.loads(report_path.read_text())
        validate_baseline_report(report, manifest)
        initial = Path(report['candidates']['two_class_epoch1']['weights'])
        initial_metrics = report['candidates']['two_class_epoch1']['metrics']
    selection_file = output / 'minimum_class_selection.json'
    initial_score = -1.0
    if args.resume:
        experiment = json.loads((output / 'experiment.json').read_text())
        if (experiment['dataset_fingerprint'] != manifest['fingerprint'] or
                experiment['validation_protocol'] != manifest['validation_protocol'] or
                experiment['epochs'] != args.epochs or experiment['target_AP50'] != args.target):
            raise ValueError('Resume must preserve dataset, epoch horizon and target')
        resume = output / 'train/weights/last_resume.pt'
        checkpoint = torch.load(resume, map_location='cpu', weights_only=False)
        validate_resume_epoch(resume, checkpoint)
        if checkpoint.get('optimizer') is None or checkpoint['epoch'] < 0:
            raise ValueError('Resume checkpoint lacks optimizer/epoch')
        if selection_file.exists():
            selection = json.loads(selection_file.read_text())
            if digest(Path(selection['weights'])) != selection['weights_sha256']:
                raise ValueError('Selected checkpoint changed')
            initial_score = selection['minimum_AP50']
        model = YOLO(str(resume))
        options = dict(resume=str(resume), device='mps')
    else:
        if output.exists():
            raise ValueError('Run exists; use --resume or choose another --model')
        model = YOLO(str(initial))
        if list(model.names.values()) not in (NAMES, NAMES + ['bottle']) or model.model.yaml.get('scale') != 'n':
            raise ValueError('Expected source YOLO11n with known class order')
        options = dict(data=str(args.data / 'data.yaml'), epochs=args.epochs, batch=4,
                       imgsz=640, device='mps', workers=0, seed=42, patience=15,
                       project=str(output), name='train', exist_ok=False, save=True,
                       cache=False, resume=False, single_cls=False, classes=None, val=True,
                       cls_remap=True, optimizer='AdamW', lr0=.001, warmup_epochs=3.)
        output.mkdir()
        save_json(output / 'experiment.json', dict(
            dataset_fingerprint=manifest['fingerprint'], validation_protocol=manifest['validation_protocol'],
            source_weights=str(initial), source_weights_sha256=digest(initial),
            baseline_metrics=initial_metrics, target_AP50=args.target, epochs=args.epochs,
            corrected_baseline_report=str(comparison_path(manifest)) if manifest.get('benchmark_revised') else None,
            training_args=options, initialization='New optimizer; preserved YOLO11n weights, native class-name remap if needed.',
            augmentation='Standard Ultralytics defaults; no bottle replay or training image removal.'))
    callback = minimum_class_callback(output, [manifest['counts']['val']['objects'][n] for n in NAMES],
                                      args.target, initial_score=initial_score, names=NAMES,
                                      minutes=args.minutes, preserve_resume=True)
    model.add_callback('on_train_start', callback.start)
    model.add_callback('on_fit_epoch_end', callback)

    def check_head(trainer):
        if list(trainer.model.names.values()) != NAMES or trainer.model.model[-1].nc != 2:
            raise ValueError('Training head was not converted to two outputs')
    model.add_callback('on_train_start', check_head)
    model.train(**options)
    verify(args.data)
    selection = json.loads(selection_file.read_text())
    if digest(Path(selection['weights'])) != selection['weights_sha256']:
        raise ValueError('Selected checkpoint no longer matches recorded score')
    evaluate(args.data, output, Path(selection['weights']), args.target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['build', 'verify', 'baseline', 'train', 'evaluate'])
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--review-files', nargs='*', type=Path, default=[REVIEWS, CORRECTIONS])
    parser.add_argument('--rebaseline', action='store_true',
                        help='Build a new benchmark with hash-bound validation corrections; re-evaluate baselines')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--target', type=float, default=.9, help='AP50 target for EACH class')
    parser.add_argument('--minutes', type=float, help='Optional budget checked after each complete epoch')
    parser.add_argument('--resume', action='store_true', help='Continue this same two-class run with optimizer state')
    parser.add_argument('--continue-from', type=Path, help='Branch from a selected checkpoint on the same corrected data')
    parser.add_argument('--lr-scale', type=float, default=1., help='New branch only: scale learning rates, preserving optimizer/EMA')
    parser.add_argument('--weights', type=Path, help='Evaluation checkpoint; defaults to best_min_ap50.pt')
    args = parser.parse_args()
    args.data, args.model = args.data.resolve(), args.model.resolve()
    try:
        if not math.isfinite(args.lr_scale) or not 0 < args.lr_scale <= 1:
            raise ValueError('--lr-scale must be in (0, 1]')
        if args.lr_scale != 1 and not args.continue_from:
            raise ValueError('Changing learning rates requires --continue-from and a new --model directory')
        if args.continue_from and args.epochs != 80:
            raise ValueError('Learning-rate branches must retain the original 80-epoch horizon')
        if (args.epochs < 1 or not math.isfinite(args.target) or not 0 < args.target <= 1 or
                args.minutes is not None and (not math.isfinite(args.minutes) or args.minutes <= 0)):
            raise ValueError('Use positive epochs/minutes and target in (0, 1]')
        if args.command == 'build':
            build(args.source, args.data, args.review_files, rebaseline=args.rebaseline)
        elif args.command == 'verify':
            verify(args.data)
        elif args.command == 'baseline':
            baseline(args.data)
        elif args.command == 'train':
            train(args)
        else:
            evaluate(args.data, args.model, args.weights or args.model / 'train/weights/best_min_ap50.pt', args.target)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
