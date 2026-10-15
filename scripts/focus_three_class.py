"""Isolated, reviewed plate/wrapper/bottle experiment: build, train, evaluate."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import time

from PIL import Image
import yaml

from experiment import check_output, label_path, validate_rows
from prepare import PROJECT_ROOT, IMAGE_SUFFIXES
from train import class_names, group_for_split
from v2 import digest, load_dataset, validate_overrides

REVIEW = PROJECT_ROOT / 'data/audits/v2_review_2026-10-12'
DATA = PROJECT_ROOT / 'data/focus-three-class-613'
MODEL = PROJECT_ROOT / 'model/focus-three-class-613'
ACCEPTED = {'corrected_and_signed_off', 'corrected_three_class_only',
            'focus_verified', 'verified_unchanged', 'previously_signed_off'}
NAMES = ['plate', 'wrapper', 'bottle']
REMAP = {1: 0, 8: 1, 9: 2}


def target_rows(rows):
    validated = validate_rows(rows, 'focus labels', class_names())
    return [str(REMAP[int(row.split()[0])]) + ' ' + ' '.join(row.split()[1:])
            for row in validated if int(row.split()[0]) in REMAP]


def capture_group(name):
    """Bind adjacent meal views and every frame from a waste source video."""
    source, original = name.split('__', 1)
    original = re.sub(r'^(train|val|valid|test)__', '', original)
    original = original.split('.rf.')[0]
    if source == 'school_lunch':
        match = re.match(r'(\d{6})', original)
        return source + '__' + match[1] if match else None
    if source == 'waste_detection_v3':
        match = re.match(r'(.+)-\d+(?:_jpg)?$', original)
        return source + '__' + match[1] if match else None
    return None


def apply_manual_reviews(records, files):
    """Apply complete scoped reviews only to the exact image and current labels."""
    index = {r['name']: r for r in records}
    seen = set()
    for path in files:
        reviews = json.loads(path.read_text())
        if not isinstance(reviews, list):
            raise ValueError('Manual review file must contain a list')
        for review in reviews:
            name = review['name']
            if name in seen:
                raise ValueError('Duplicate manual review: ' + name)
            seen.add(name)
            if name not in index:
                raise ValueError('Unknown manual review image: ' + name)
            record = index[name]
            row_hash = hashlib.sha256(json.dumps(record['rows'], sort_keys=True).encode()).hexdigest()
            if review.get('image_sha256') != record['sha'] or review.get('input_rows_sha256') != row_hash:
                raise ValueError('Manual review hash mismatch: ' + name)
            status = review.get('status')
            if status not in ('verified_unchanged', 'corrected', 'unresolved'):
                raise ValueError('Invalid manual review status: ' + name)
            if not review.get('reviewer') or not review.get('reason'):
                raise ValueError('Manual review requires reviewer and reason: ' + name)
            record['review_status'] = status
            record['review_reason'] = review['reason']
            record['accepted'] = status != 'unresolved'
            if status != 'unresolved':
                if not isinstance(review.get('rows'), list):
                    raise ValueError('Manual review rows must be a list: ' + name)
                rows = validate_rows(review['rows'], name, NAMES)
                if rows != review['rows'] or len(set(rows)) != len(rows):
                    raise ValueError('Invalid or duplicate manual rows: ' + name)
                if status == 'verified_unchanged' and rows != record['rows']:
                    raise ValueError('Unchanged review changed rows: ' + name)
                record['rows'] = rows
                record['verified_absence'] = not rows
    return records


def repair_groups(records):
    """Unite original-photo, byte and decoded-pixel groups; minimize moves."""
    parent = list(range(len(records)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for field in ('group', 'sha', 'pixels', 'capture'):
        seen = {}
        for i, record in enumerate(records):
            key = record.get(field)
            if key is None:
                continue
            if key in seen:
                parent[root(i)] = root(seen[key])
            seen[key] = i
    components = {}
    for i in range(len(records)):
        components.setdefault(root(i), []).append(records[i])
    moves = []
    for members in components.values():
        counts = Counter(r['split'] for r in members)
        chosen = ('val' if any(r.get('frozen_val') for r in members) else
                  max(('train', 'val'), key=lambda s: (counts[s], s == 'train')))
        for record in members:
            if record['split'] != chosen:
                moves.append(dict(image=record['name'], old=record['split'], new=chosen))
                record['split'] = chosen
    return moves


def audit(records, expected_count=613):
    summary = {s: dict(images=0, negatives=0, objects=dict.fromkeys(NAMES, 0),
                       class_images=dict.fromkeys(NAMES, 0)) for s in ('train', 'val')}
    seen = {field: {} for field in ('group', 'sha', 'pixels', 'capture')}
    for r in records:
        for field in seen:
            key = r.get(field)
            if key is None:
                continue
            prior = seen[field].setdefault(key, r['split'])
            if prior != r['split']:
                raise ValueError(f"Split leakage ({field}): {r['name']}")
        if r.get('frozen_val') and r['split'] != 'val':
            raise ValueError('Frozen validation anchor moved: ' + r['name'])
        rows = validate_rows(r['rows'], r['name'], NAMES)
        if rows != r['rows']:
            raise ValueError('Unexpected annotation normalization')
        if not rows and not r['verified_absence']:
            raise ValueError('Unverified negative: ' + r['name'])
        stats = summary[r['split']]
        stats['images'] += 1
        stats['negatives'] += not rows
        ids = [int(row.split()[0]) for row in rows]
        for i, name in enumerate(NAMES):
            stats['objects'][name] += ids.count(i)
            stats['class_images'][name] += i in ids
    if len(records) != expected_count or len({r['name'] for r in records}) != expected_count:
        raise ValueError(f'Expected exactly {expected_count} unique accepted images')
    if any(not stats['images'] or not all(stats['objects'].values()) for stats in summary.values()):
        raise ValueError('Each split must contain all three classes')
    return summary


def pixels(image):
    with Image.open(image) as opened:
        rgb = opened.convert('RGB')
        return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def fingerprint(records):
    sha = hashlib.sha256()
    for r in sorted(records, key=lambda r: r['name']):
        evidence = {k: r[k] for k in (
            'name', 'split', 'sha', 'rows', 'group', 'pixels', 'verified_absence',
            'review_status', 'original_split', 'input_label_sha256', 'override_applied')}
        for key in ('capture', 'frozen_val'):
            if key in r:
                evidence[key] = r[key]
        sha.update(json.dumps(evidence, sort_keys=True).encode())
    return sha.hexdigest()


def build(args):
    source = args.source.resolve()
    output = args.data.resolve()
    check_output(output, [source.parent, PROJECT_ROOT / 'data/training'])
    sources = load_dataset(source)
    index = {}
    for split, images in sources.items():
        for image in images:
            if image.name in index:
                raise ValueError('Duplicate V2 filename: ' + image.name)
            index[image.name] = (split, image)
    reviews = list(csv.DictReader((REVIEW / 'plate_wrapper_bottle_focus_review.csv').open()))
    selected = [r for r in reviews if r['review_result'] in ACCEPTED]
    if len(reviews) != 803 or len(selected) != 613:
        raise ValueError('Review selection must be 613 accepted and 190 excluded')
    # Overrides validate their input hashes before being applied to isolated copies.
    overrides = {r['stem']: r for r in validate_overrides(REVIEW / 'reviewed_overrides.json', sources)}
    unchanged = {(r['source'], r['image']): r for r in csv.DictReader((REVIEW / 'image_review.csv').open())
                 if r.get('decision') == 'checked_existing' and r.get('review_scope') == 'full_resolution'}
    records = []
    for review in selected:
        if review['image_viewed'] != 'yes' or set(review['scope'].split('|')) != set(NAMES):
            raise ValueError('Missing three-class visual verification: ' + str(review))
        name = review['source'] + '__' + review['image']
        if name not in index:
            raise ValueError('Accepted image missing from V2: ' + name)
        split, image = index[name]
        if split not in ('train', 'val'):
            raise ValueError('Unexpected V2 split: ' + split)
        label = label_path(image)
        rows = [r for r in label.read_text().splitlines() if r.strip()]
        override = overrides.get(image.stem)
        if override:
            if override['mode'] == 'replace':
                rows = override['rows']
            else:
                if set(rows) & set(override['rows']):
                    raise ValueError('Duplicate appended box: ' + name)
                rows += override['rows']
        elif review['review_result'].startswith('corrected'):
            raise ValueError('Correction missing approved override: ' + name)
        if not override and (review['source'], review['image']) in unchanged:
            expected = unchanged[(review['source'], review['image'])]['label_sha256']
            if digest(label) != expected:
                raise ValueError('Verified unchanged label hash mismatch: ' + name)
        target = target_rows(rows)
        # An empty file alone cannot prove absence. Require an approved full
        # replacement or an unchanged label tied to full-resolution sign-off.
        absence_verified = not target and (
            (override is not None and override['mode'] == 'replace') or
            (not override and (review['source'], review['image']) in unchanged))
        original = group_for_split(review['source'], review['image']).lower()
        records.append(dict(name=name, split=split, original_split=split,
                            group=review['source'] + '__' + original,
                            sha=digest(image), pixels=pixels(image), rows=target,
                            verified_absence=absence_verified, review_status=review['review_result'],
                            image=str(image), input_label_sha256=digest(label),
                            override_applied=bool(override)))
    moves = repair_groups(records)
    summary = audit(records)
    manifest = dict(selected=613, excluded=190, source=str(source),
                    review_sha256=digest(REVIEW / 'plate_wrapper_bottle_focus_review.csv'),
                    overrides_sha256=digest(REVIEW / 'reviewed_overrides.json'),
                    counts=summary, split_moves=moves, records=records,
                    fingerprint=fingerprint(records), leakage='passed: original, byte, decoded-pixel groups',
                    negative_policy='Accepted full three-class visual review verifies target absence')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'records'}, indent=2), flush=True)
    for split in ('train', 'val'):
        (output / 'images' / split).mkdir(parents=True)
        (output / 'labels' / split).mkdir(parents=True)
    for r in records:
        shutil.copy2(r['image'], output / 'images' / r['split'] / r['name'])
        (output / 'labels' / r['split'] / Path(r['name']).with_suffix('.txt')).write_text(
            '\n'.join(r['rows']) + ('\n' if r['rows'] else ''))
    (output / 'data.yaml').write_text(yaml.safe_dump(dict(path=str(output), train='images/train',
                                                        val='images/val', names=NAMES), sort_keys=False))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    verify(output)


def validate_anchors(records, baseline, allow_plate_relabel=False):
    """Keep frozen photographs fixed; explicitly record plate-only rebaselining."""
    current = {r['name']: r for r in records}
    changes = []
    for anchor in baseline['records']:
        if anchor['split'] != 'val':
            continue
        record = current.get(anchor['name'])
        if record is None:
            raise ValueError('Frozen validation anchor lacks current accepted review: ' + anchor['name'])
        if any(record[key] != anchor[key] for key in ('sha', 'pixels')):
            raise ValueError('Frozen validation evidence changed: ' + anchor['name'])
        if record['rows'] != anchor['rows']:
            nonplate = lambda rows: sorted(row for row in rows if row.split()[0] != '0')
            if not allow_plate_relabel or nonplate(record['rows']) != nonplate(anchor['rows']):
                raise ValueError('Frozen validation evidence changed: ' + anchor['name'])
            changes.append(dict(name=anchor['name'], original_rows=anchor['rows'], new_rows=record['rows']))
    return changes


def build_all(args):
    """Build a separate expanded pool from original V2 and current review evidence."""
    source, output = args.source.resolve(), args.data.resolve()
    if source != (PROJECT_ROOT / 'data/training-v2/data.yaml').resolve():
        raise ValueError('build_all requires original data/training-v2/data.yaml')
    check_output(output, [source.parent, PROJECT_ROOT / 'data/training', DATA])
    sources = load_dataset(source)
    overrides = {r['stem']: r for r in validate_overrides(REVIEW / 'reviewed_overrides.json', sources)}
    focus = {r['source'] + '__' + r['image']: r for r in csv.DictReader(
        (REVIEW / 'plate_wrapper_bottle_focus_review.csv').open())}
    unchanged = {r['source'] + '__' + r['image']: r for r in csv.DictReader(
        (REVIEW / 'image_review.csv').open()) if r.get('decision') == 'checked_existing'
        and r.get('review_scope') == 'full_resolution'}
    baseline = verify(DATA)
    anchor_rows = {r['name']: r['rows'] for r in baseline['records'] if r['split'] == 'val'}
    anchors = set(anchor_rows)
    if len(anchors) != 92:
        raise ValueError('Expected 92 frozen baseline validation anchors')
    records = []
    for split, images in sources.items():
        for image in images:
            label = label_path(image)
            rows = [r for r in label.read_text().splitlines() if r.strip()]
            override = overrides.get(image.stem)
            if override:
                if override['mode'] == 'replace':
                    rows = override['rows']
                else:
                    if set(rows) & set(override['rows']):
                        raise ValueError('Duplicate appended box: ' + image.name)
                    rows += override['rows']
            ledger = unchanged.get(image.name)
            if ledger and not override and digest(label) != ledger['label_sha256']:
                raise ValueError('Verified unchanged label hash mismatch: ' + image.name)
            review = focus.get(image.name, {})
            scoped = (review.get('review_result') in ACCEPTED and review.get('image_viewed') == 'yes'
                      and set(review.get('scope', '').split('|')) == set(NAMES))
            if scoped and review['review_result'].startswith('corrected') and not override:
                raise ValueError('Correction missing approved override: ' + image.name)
            full = bool((override and override['mode'] == 'replace') or ledger)
            target = target_rows(rows)
            accepted = scoped or full
            origin, original = image.name.split('__', 1)
            record = dict(name=image.name, split=split, original_split=split,
                          group=origin + '__' + group_for_split(origin, original).lower(),
                          sha=digest(image), pixels=pixels(image), rows=target,
                          verified_absence=accepted and not target,
                          review_status=review.get('review_result', 'full_verified' if full else 'pending'),
                          review_reason=review.get('evidence_and_remaining_work', 'No complete target review'),
                          image=str(image), input_label_sha256=digest(label),
                          override_applied=bool(override), accepted=accepted,
                          capture=capture_group(image.name), frozen_val=image.name in anchors)
            records.append(record)
    apply_manual_reviews(records, args.review_files)
    selected = [r for r in records if r['accepted']]
    pending = [r for r in records if not r['accepted']]
    if not anchors <= {r['name'] for r in selected}:
        raise ValueError('Frozen validation anchor lacks current accepted review')
    label_changes = validate_anchors(selected, baseline,
                                     allow_plate_relabel=args.one_box_per_stack)
    # Include held photos in grouping so an unresolved frame can still connect two
    # accepted views of the same capture. No held labels enter the dataset.
    moves = repair_groups(records)
    summary = audit(selected, expected_count=len(selected))
    manifest = dict(selected=len(selected), excluded=len(pending), source=str(source),
                    build_mode='all_verified', review_sha256=digest(REVIEW / 'plate_wrapper_bottle_focus_review.csv'),
                    overrides_sha256=digest(REVIEW / 'reviewed_overrides.json'),
                    manual_review_files=[dict(path=str(p.resolve()), sha256=digest(p)) for p in args.review_files],
                    frozen_validation_anchors=sorted(anchors), counts=summary, split_moves=moves,
                    records=selected, fingerprint=fingerprint(selected),
                    leakage='passed: original, byte, decoded-pixel and capture groups',
                    negative_policy='Complete visual target review required; unresolved photos held')
    if args.one_box_per_stack:
        manifest.update(plate_stack_policy='one_box_per_stack',
                        validation_label_changes=label_changes,
                        needs_rebaseline=bool(label_changes),
                        benchmark_note='Re-evaluate baseline weights on these labels before comparing scores.'
                        if label_changes else 'Frozen validation labels unchanged.')
    for split in ('train', 'val'):
        (output / 'images' / split).mkdir(parents=True)
        (output / 'labels' / split).mkdir(parents=True)
    for record in selected:
        shutil.copy2(record['image'], output / 'images' / record['split'] / record['name'])
        (output / 'labels' / record['split'] / Path(record['name']).with_suffix('.txt')).write_text(
            '\n'.join(record['rows']) + ('\n' if record['rows'] else ''))
    (output / 'data.yaml').write_text(yaml.safe_dump(dict(path=str(output), train='images/train',
                                                        val='images/val', names=NAMES), sort_keys=False))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'pending_review.json').write_text(json.dumps(pending, indent=2) + '\n')
    verify(output)
    print(f'Held unresolved: {len(pending)} photos ({sum(bool(r["rows"]) for r in pending)} target-positive).')


def verify(data):
    manifest = json.loads((data / 'manifest.json').read_text())
    spec = yaml.safe_load((data / 'data.yaml').read_text())
    if spec != dict(path=str(data.resolve()), train='images/train', val='images/val', names=NAMES):
        raise ValueError('Dataset YAML changed')
    records = manifest['records']
    if 'frozen_validation_anchors' in manifest:
        anchors = {r['name'] for r in records if r.get('frozen_val')}
        if anchors != set(manifest['frozen_validation_anchors']) or len(anchors) != 92:
            raise ValueError('Frozen validation anchor membership changed')
    for split in ('train', 'val'):
        expected = {r['name'] for r in records if r['split'] == split}
        actual = {p.name for p in (data / 'images' / split).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
        labels = {p.name for p in (data / 'labels' / split).glob('*.txt')}
        if actual != expected or labels != {str(Path(n).with_suffix('.txt')) for n in expected}:
            raise ValueError('Dataset image/label membership changed')
    for r in records:
        image = data / 'images' / r['split'] / r['name']
        rows = label_path(image).read_text().splitlines()
        if digest(image) != r['sha'] or rows != r['rows']:
            raise ValueError('Dataset changed after build: ' + r['name'])
        source, original = r['name'].split('__', 1)
        group = source + '__' + group_for_split(source, original).lower()
        if 'capture' in r and capture_group(r['name']) != r['capture']:
            raise ValueError('Dataset capture evidence changed: ' + r['name'])
        if group != r['group'] or pixels(image) != r['pixels']:
            raise ValueError('Dataset leakage evidence changed: ' + r['name'])
    if (fingerprint(records) != manifest['fingerprint'] or
            audit(records, expected_count=manifest['selected']) != manifest['counts']):
        raise ValueError('Dataset manifest integrity failure')
    print(f"Integrity passed: {manifest['selected']} images, validated boxes, verified negatives, no split leakage.", flush=True)
    print(json.dumps(manifest['counts'], indent=2), flush=True)
    return manifest


def minimum_class_callback(output, expected_counts, target, initial_score=-1.0,
                           minutes=None, preserve_resume=False):
    """Keep the strongest weakest class, binding each score to that epoch's EMA."""
    best = initial_score
    started = None

    def start(trainer):
        nonlocal started
        started = time.monotonic()

    def select(trainer):
        nonlocal best
        # This hook also runs during final revalidation of a different checkpoint.
        with Path(trainer.csv).open() as stream:
            history = list(csv.DictReader(stream))
        if not history or int(history[-1]['epoch']) != trainer.epoch + 1:
            return
        if preserve_resume:
            latest = output / 'train/weights/last_resume.pt'
            latest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(trainer.last, latest)
        if minutes is not None:
            if started is None:
                start(trainer)
            if time.monotonic() - started >= minutes * 60:
                trainer.stop = True
        metrics = trainer.validator.metrics
        indices = [int(i) for i in metrics.box.ap_class_index]
        if (set(indices) != set(range(3)) or len(indices) != 3 or
                list(map(int, metrics.nt_per_class)) != expected_counts):
            raise ValueError('Per-class checkpoint selection has incorrect validation support')
        scores = {NAMES[c]: float(metrics.box.ap50[j]) for j, c in enumerate(indices)}
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values()):
            raise ValueError('Invalid per-class checkpoint metrics')
        score = min(scores.values())
        if score > best:
            location = output / 'train/weights/best_min_ap50.pt'
            location.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(trainer.last, location)
            best = score
            (output / 'minimum_class_selection.json').write_text(json.dumps(dict(
                epoch=trainer.epoch + 1, per_class_AP50=scores, minimum_AP50=score,
                target_AP50=target, weights=str(location), weights_sha256=digest(location)
            ), indent=2) + '\n')
        print('Per-class AP50:', scores, 'minimum:', score, flush=True)
        if score >= target:
            trainer.stop = True

    select.start = start
    return select


def profile_options(path, data, manifest):
    """Validate a recipe's training subset against the audited image pool."""
    recipe = json.loads(path.read_text())
    options = dict(recipe['training_args'])
    if options.get('cfg') is not None:
        raise ValueError('Recipe cfg can hide protected settings; specify training_args directly')
    options.pop('cfg', None)
    if options.get('resume', False):
        raise ValueError('Recipe resume could overwrite an earlier run; use a fresh experiment')
    options.pop('save_dir', None)  # Only the new CLI --model directory may receive writes.
    weights = options.pop('model')
    config = Path(options['data']).resolve()
    spec = yaml.safe_load(config.read_text())
    if (spec['names'] != NAMES or Path(spec['path']).resolve() != data or
            (data / spec['val']).resolve() != data / 'images/val'):
        raise ValueError('Recipe must use audited classes and unchanged validation')
    training = (data / spec['train']).resolve()
    if training.is_file():
        known = {data / 'images/train' / r['name'] for r in manifest['records'] if r['split'] == 'train'}
        paths = [Path(row).resolve() for row in training.read_text().splitlines() if row.strip()]
        if not paths or set(paths) - known:
            raise ValueError('Recipe contains unknown or validation training images')
    elif training != data / 'images/train':
        raise ValueError('Recipe training folder is outside the audited pool')
    if (options.get('single_cls', False) or options.get('classes') is not None or
            options.get('split', 'val') != 'val' or options.get('val', True) is not True):
        raise ValueError('Recipe must evaluate all three classes on validation')
    target = float(recipe.get('target_min_ap50', .8))
    if not 0 < target <= 1:
        raise ValueError('Invalid per-class AP50 target')
    return weights, options, target


def train(args):
    manifest = verify(args.data)
    from ultralytics import YOLO
    import torch
    if not torch.backends.mps.is_available():
        raise ValueError('MPS unavailable; refusing device substitution')
    if args.model.exists():
        raise ValueError('Model output exists; choose a separate --model directory')
    pretrained = PROJECT_ROOT / 'model/weights/yolo11n.pt'
    weights = str(pretrained) if pretrained.is_file() else 'yolo11n.pt'
    # Default experiment retains standard augmentation; recipes are explicit alternatives.
    options = dict(data=str(args.data / 'data.yaml'), epochs=args.epochs, batch=4,
                   imgsz=640, workers=0, seed=42, patience=0)
    target = None
    if args.recipe:
        weights, options, target = profile_options(args.recipe, args.data, manifest)
    options.update(device='mps', project=str(args.model), name='train', exist_ok=False, save=True,
                   resume=False, single_cls=False, classes=None, val=True, split='val', cfg=None)
    model = YOLO(weights)
    if target is not None:
        args.model.mkdir(parents=True)
        (args.model / 'experiment.json').write_text(json.dumps(dict(
            training_args={**options, 'model': weights}, target_min_ap50=target,
            dataset_fingerprint=manifest['fingerprint']), indent=2) + '\n')
        model.add_callback('on_fit_epoch_end', minimum_class_callback(
            args.model, [manifest['counts']['val']['objects'][n] for n in NAMES], target))
    model.train(**options)
    completed = list(csv.DictReader((args.model / 'train/results.csv').open()))
    if not completed:
        raise ValueError('No training epoch completed')
    if not args.recipe and len(completed) != args.epochs:
        raise ValueError(f'Incomplete training: {len(completed)}/{args.epochs} epochs')


def continuation_output(output, source_experiment):
    """Allow only fresh experiment children of the isolated improvement root."""
    allowed = (PROJECT_ROOT / 'model/focus-three-class-improvement').resolve()
    if output.exists() or output.is_symlink():
        raise ValueError('Continuation output already exists: ' + str(output))
    resolved = output.resolve()
    source = source_experiment.resolve()
    if resolved.parent != allowed or resolved == source or source in resolved.parents:
        raise ValueError('Continuation output must be a new direct child of ' + str(allowed))


def continuation_checkpoint(checkpoint, output, data, epochs):
    """Change only the cloned checkpoint's run arguments, retaining optimizer state."""
    epoch = checkpoint.get('epoch')
    if (not isinstance(checkpoint.get('optimizer'), dict) or not isinstance(epoch, int)
            or epoch < 0 or epochs <= epoch + 1):
        raise ValueError('Continuation needs an unstripped optimizer and unfinished total epoch horizon')
    if not isinstance(checkpoint.get('train_args'), dict):
        raise ValueError('Continuation checkpoint lacks training arguments')
    clone = dict(checkpoint)
    clone['train_args'] = {**checkpoint['train_args'], 'epochs': epochs, 'time': None,
                          'data': str(data / 'data.yaml'), 'project': str(output), 'name': 'train',
                          'save_dir': str(output / 'train'), 'device': 'mps',
                          'resume': str(output / 'incoming.pt'), 'single_cls': False,
                          'classes': None, 'val': True, 'split': 'val', 'cfg': None}
    return clone


def continue_train(args):
    """Continue optimizer learning in a separate run without a time-compressed schedule."""
    manifest = verify(args.data)
    if not args.continue_from:
        raise ValueError('continue_train requires --continue-from')
    source = args.continue_from.resolve()
    parent = source.parent.parent.parent
    if source.parent != parent / 'train/weights' or not source.is_file():
        raise ValueError('Continuation checkpoint must be inside source experiment/train/weights')
    experiment = json.loads((parent / 'experiment.json').read_text())
    if experiment.get('dataset_fingerprint') != manifest['fingerprint']:
        raise ValueError('Continuation source dataset fingerprint mismatch')
    continuation_output(args.model, parent)
    from ultralytics import YOLO
    import torch
    if not torch.backends.mps.is_available():
        raise ValueError('MPS unavailable; refusing device substitution')
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    model = checkpoint.get('ema')
    if model is None:
        model = checkpoint.get('model')
    names = getattr(model, 'names', None)
    if (list(names.values()) if isinstance(names, dict) else names) != NAMES:
        raise ValueError('Continuation checkpoint class names mismatch')
    clone = continuation_checkpoint(checkpoint, args.model, args.data, args.epochs)
    # Validate inherited best before creating any output or launching a trainer.
    selection_path = parent / 'minimum_class_selection.json'
    inherited = None
    if selection_path.is_file():
        inherited = json.loads(selection_path.read_text())
        best = parent / 'train/weights/best_min_ap50.pt'
        if digest(best) != inherited.get('weights_sha256'):
            raise ValueError('Inherited best checkpoint hash mismatch')
        score = inherited.get('minimum_AP50')
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid inherited best score')
    args.model.mkdir(parents=True)
    incoming = args.model / 'incoming.pt'
    torch.save(clone, incoming)
    if inherited is not None:
        location = args.model / 'train/weights/best_min_ap50.pt'
        location.parent.mkdir(parents=True)
        shutil.copy2(best, location)
        inherited.update(weights=str(location), weights_sha256=digest(location), target_AP50=.9)
        (args.model / 'minimum_class_selection.json').write_text(json.dumps(inherited, indent=2) + '\n')
    (args.model / 'experiment.json').write_text(json.dumps(dict(
        dataset_fingerprint=manifest['fingerprint'], source_checkpoint=str(source),
        source_checkpoint_sha256=digest(source), source_epoch=checkpoint['epoch'],
        original_args=checkpoint['train_args'], training_args=clone['train_args'],
        total_epoch_horizon=args.epochs, wall_limit_minutes=args.minutes, target_min_ap50=.9,
        continuation_note='Retains optimizer learning; earlier time-compressed epochs are not an exact uninterrupted 80-epoch run.'
    ), indent=2) + '\n')
    callback = minimum_class_callback(args.model,
        [manifest['counts']['val']['objects'][n] for n in NAMES], .9,
        initial_score=inherited['minimum_AP50'] if inherited else -1,
        minutes=args.minutes, preserve_resume=True)
    continued = YOLO(str(incoming))
    continued.add_callback('on_train_start', callback.start)
    continued.add_callback('on_fit_epoch_end', callback)
    # Installed check_resume restores checkpoint args; cloned save_dir/project/name
    # therefore isolate writes, and its allowlist accepts save_dir and time again.
    continued.train(resume=str(incoming), time=None, save_dir=str(args.model / 'train'),
                    data=str(args.data / 'data.yaml'), device='mps')


def evaluate(args):
    manifest = verify(args.data)
    from ultralytics import YOLO
    weights = args.eval_weights or args.model / 'train/weights/best.pt'
    if (args.model / args.eval_name).exists():
        raise ValueError('Evaluation output already exists')
    model = YOLO(str(weights))
    if list(model.names.values()) != NAMES:
        raise ValueError('Checkpoint class names mismatch')
    metrics = model.val(data=str(args.data / 'data.yaml'), split='val', device='mps',
                        batch=4, imgsz=args.eval_imgsz, workers=0, project=str(args.model),
                        name=args.eval_name, plots=True, augment=args.eval_augment)
    box = metrics.box
    indices = [int(i) for i in box.ap_class_index]
    if set(indices) != set(range(3)) or len(indices) != 3:
        raise ValueError('Evaluation missing class metrics')
    if list(map(int, metrics.nt_per_class)) != [manifest['counts']['val']['objects'][n] for n in NAMES]:
        raise ValueError('Evaluated object support differs from audited dataset')
    per_class = {NAMES[i]: dict(precision=float(box.p[j]), recall=float(box.r[j]),
                               AP50=float(box.ap50[j]), AP50_95=float(box.ap[j]),
                               objects=manifest['counts']['val']['objects'][NAMES[i]])
                 for j, i in enumerate(indices)}
    overall = dict(precision=float(box.mp), recall=float(box.mr), AP50=float(box.map50), AP50_95=float(box.map))
    if any(not math.isfinite(v) or not 0 <= v <= 1 for r in [overall, *per_class.values()]
           for k, v in r.items() if k != 'objects'):
        raise ValueError('Invalid evaluation metrics')
    result = dict(overall=overall, per_class=per_class, weights=str(weights),
                  weights_sha256=digest(weights), dataset_fingerprint=manifest['fingerprint'],
                  imgsz=args.eval_imgsz, augment=args.eval_augment,
                  interpretation='Reviewed public validation; no independent test or improvement claim.')
    (args.model / args.eval_name / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['build', 'build_all', 'train', 'continue_train', 'evaluate'])
    parser.add_argument('--continue-from', type=Path, help='Unstripped checkpoint in an earlier experiment')
    parser.add_argument('--minutes', type=float, default=9, help='Continuation wall limit checked after complete epochs')
    parser.add_argument('--source', type=Path, default=PROJECT_ROOT / 'data/training-v2/data.yaml')
    parser.add_argument('--data', type=Path, default=None)
    parser.add_argument('--one-box-per-stack', action='store_true',
                        help='build_all: allow documented plate-only validation relabeling; requires rebaseline')
    parser.add_argument('--review-files', nargs='*', type=Path, default=[],
                        help='Hash-bound complete manual three-class reviews for build_all')
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--recipe', type=Path, help='Explicit training recipe with weakest-class checkpoint selection')
    parser.add_argument('--epochs', type=int, default=80,
                        help='Training epochs (use 10 for the revised short experiment)')
    parser.add_argument('--eval-imgsz', type=int, default=640,
                        help='Evaluation image size (default 640)')
    parser.add_argument('--eval-name', default='evaluation', help='Separate evaluation output name')
    parser.add_argument('--eval-augment', action='store_true',
                        help='Evaluate native multi-scale and flip inference')
    parser.add_argument('--eval-weights', type=Path,
                        help='Explicit checkpoint for evaluating last or per-class-selected weights')
    args = parser.parse_args()
    args.data = (args.data or (PROJECT_ROOT / 'data/focus-three-class-expanded'
                             if args.command == 'build_all' else DATA)).resolve()
    args.model = args.model.resolve()
    try:
        if not math.isfinite(args.minutes) or args.minutes <= 0:
            raise ValueError('--minutes must be positive and finite')
        if args.epochs < 1:
            raise ValueError('--epochs must be positive')
        if args.eval_imgsz < 32 or args.eval_imgsz % 32:
            raise ValueError('--eval-imgsz must be a positive multiple of 32')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', args.eval_name):
            raise ValueError('--eval-name must contain only letters, digits, dashes or underscores')
        globals()[args.command](args)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
