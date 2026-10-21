"""Build/verify an isolated, capture-grouped recovery benchmark with V2 labels.

Keeps all 11 IDs and all source images. Never reuses an old trained checkpoint
on newly held-out captures. Reuses the repository's annotation, coverage-loss,
checkpoint and evaluation helpers; the original training pipeline is unchanged.
"""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
import shutil
import time

import yaml

from experiment import box_iou, check_output, label_path, prepare_weak_subset, validate_rows
from focus_three_class import (atomic_checkpoint_copy, capture_group, pixels, repair_groups,
                               continuation_checkpoint, validate_image_file, validate_resume_epoch)
from prepare import IMAGE_SUFFIXES, PROJECT_ROOT
from train import class_names
from v2 import dataset_fingerprint, digest, load_dataset, source_group

SOURCE = PROJECT_ROOT / 'data/training-v2-codex-reviewed/data.yaml'
DATA = PROJECT_ROOT / 'data/utensil-focus/grouped-v2'
REVIEWS = PROJECT_ROOT / 'data/utensil-focus/utensil-additions.json'
VIEWS = PROJECT_ROOT / 'data/utensil-focus/weak-v2-clean/data.yaml'
OUTPUT = PROJECT_ROOT / 'model/utensil-focus/recovery80-clean'
POLICY = PROJECT_ROOT / 'data/audits/model_revisions_2026-10-12/source_coverage.json'
HOLDS = PROJECT_ROOT / 'data/utensil-focus/training-holds.json'


def require_local(paths):
    offloaded = [str(p) for p in paths if getattr(Path(p).stat(), 'st_flags', 0) & 0x40000000]
    if offloaded:
        raise ValueError(f'iCloud offloaded {len(offloaded)} required file(s); use Keep Downloaded before retrying. First: {offloaded[0]}')


def require_local_dataset(source):
    require_local([source])
    photos = [p for images in load_dataset(source).values() for p in images]
    require_local([p for image in photos for p in (image, label_path(image))])


def capture(name):
    # Full-resolution/contact-sheet review found repeated household backgrounds
    # across these dates too. Keep the entire set together, not just each date.
    if name.startswith('utensil__'):
        if 'WhatsApp-Image-2022-' in name or re.search(r'__2022\d{4}_', name):
            return 'utensil__household_scenes'
        if 'IMG_20231109_' in name:
            return 'utensil__tray_20231109'
    if name.startswith('tartanhacks__'):
        return 'tartanhacks__heldout_source'
    if name.startswith('condiments__'):
        return 'condiments__same_background_batch'
    return capture_group(str(Path(name).with_suffix('')))


def project(source, reviews, external=None):
    require_local_dataset(source)
    require_local([reviews])
    names = class_names()
    records, paths = [], {}
    for split, images in load_dataset(source).items():
        if split not in ('train', 'val') and images:
            raise ValueError('Recovery must not consume an independent test set')
        for image in images:
            if image.name in paths:
                raise ValueError('Duplicate source image name')
            validate_image_file(image)
            label = label_path(image)
            rows = validate_rows(label.read_text().splitlines(), label, names)
            paths[image.name] = image
            records.append(dict(name=image.name, split=split, original_split=split,
                sha=digest(image), pixels=pixels(image), group=source_group(image.name),
                capture=capture(image.name), rows=rows, input_label_sha256=digest(label),
                frozen_val=split == 'val' or image.name.startswith('tartanhacks__')))
    index = {r['name']: r for r in records}
    seen = set()
    changes = json.loads(reviews.read_text())
    if not isinstance(changes, list):
        raise ValueError('Review file must contain a list')
    for review in changes:
        record = index.get(review.get('name'))
        if record is None or record['name'] in seen:
            raise ValueError('Missing or duplicate review image')
        seen.add(record['name'])
        if (review.get('image_sha256') != record['sha'] or
                review.get('input_label_sha256') != record['input_label_sha256']):
            raise ValueError('Review hash mismatch')
        if not review.get('reviewer') or not review.get('reason'):
            raise ValueError('Review requires reviewer and visual evidence')
        removed = validate_rows(review.get('remove_rows', []), record['name'], names)
        if (len(set(removed)) != len(removed) or any(row not in record['rows'] or
                int(row.split()[0]) not in (3, 5, 7) for row in removed)):
            raise ValueError('Training correction must name existing recovery-class rows exactly')
        for row in removed:
            record['rows'].remove(row)
        additions = validate_rows(review['additions'], record['name'], names)
        for row in additions:
            if int(row.split()[0]) not in (3, 5, 7):
                raise ValueError('Only reviewed recovery-class additions are allowed')
            if any(existing.split()[0] == row.split()[0] and box_iou(existing, row) > .85
                   for existing in record['rows']):
                raise ValueError('Duplicate recovery-class addition')
            record['rows'].append(row)
        removed_classes = {int(row.split()[0]) for row in removed}
        now_absent = removed_classes - {int(row.split()[0]) for row in record['rows']}
        if not now_absent <= set(review.get('verified_absent', [])):
            raise ValueError('Removing the last target requires explicit verified absence, not an assumed negative')
    if external:
        require_local([external])
        identities = {field: {r[field] for r in records} for field in ('sha', 'pixels', 'group')}
        old_captures = {r['capture'] for r in records}
        for review in json.loads(Path(external).read_text()):
            name = review.get('name', '')
            image = Path(review['image']).resolve()
            if (not re.fullmatch(r'packet_extra__[a-zA-Z0-9_]+\.jpg', name) or name in paths
                    or review.get('split') not in ('train', 'val')
                    or not all(review.get(k) for k in ('capture', 'original_group', 'reviewer',
                                                       'reason', 'url', 'license', 'artist'))
                    or review.get('exhaustive_packet_review') is not True):
                raise ValueError('External packets require unique names, capture, license and complete packet review')
            if review['split'] == 'val' and review.get('verified_utensil_absence') is not True:
                raise ValueError('Packet-only validation imports require verified utensil absence; missing utensil labels would bias joint AP')
            if review['capture'] in old_captures:
                raise ValueError('New packet capture overlaps an existing capture')
            require_local([image])
            validate_image_file(image)
            if digest(image) != review.get('image_sha256'):
                raise ValueError('External image hash mismatch: ' + name)
            identity = dict(sha=digest(image), pixels=pixels(image), group=review['original_group'])
            if any(identity[field] in identities[field] for field in identities):
                raise ValueError('External import overlaps an existing or duplicate original photo: ' + name)
            for field in identities:
                identities[field].add(identity[field])
            rows = validate_rows(review['rows'], name, names)
            if not rows or any(int(row.split()[0]) != 7 for row in rows):
                raise ValueError('External import requires reviewed positive sauce sachets only')
            cutlery = validate_rows(review.get('utensil_additions', []), name, names)
            if cutlery and (review['split'] != 'train' or
                    review.get('exhaustive_utensil_review') is not True or
                    any(int(row.split()[0]) != 3 for row in cutlery)):
                raise ValueError('Extra eating cutlery needs exhaustive training-photo review')
            rows += cutlery
            split = review['split']
            records.append(dict(name=name, split=split, original_split=split,
                sha=digest(image), pixels=pixels(image), group=review['original_group'],
                capture=review['capture'], rows=rows, input_label_sha256=None,
                frozen_val=split == 'val', external_provenance=review))
            paths[name] = image
    moves = repair_groups(records)
    corrected = {r['name'] for r in changes if r.get('remove_rows')}
    if any(r['name'] in corrected and r['split'] != 'train' for r in records):
        raise ValueError('Box replacement is restricted to training; validation labels remain fixed')
    # Never silently repair a reviewed external split. Related captures must be
    # assigned together in the review before the dataset can be built.
    if any(r.get('external_provenance') and r['split'] != r['original_split'] for r in records):
        raise ValueError('External review has capture/hash/original-image split overlap')
    return records, paths, moves


def audit(records):
    names = class_names()
    counts = {s: dict.fromkeys(names, 0) for s in ('train', 'val')}
    seen = {field: {} for field in ('group', 'sha', 'pixels', 'capture')}
    for record in records:
        split = record['split']
        for field, values in seen.items():
            key = record.get(field)
            if key is not None and values.setdefault(key, split) != split:
                raise ValueError('Split leakage: ' + field)
        if record['frozen_val'] and split != 'val':
            raise ValueError('Validation anchor moved to training')
        for row in validate_rows(record['rows'], record['name'], names):
            counts[split][names[int(row.split()[0])]] += 1
    if any(counts[s][name] == 0 for s in ('train', 'val') for name in ('utensil', 'napkin')):
        raise ValueError('Utensil and napkin need positive training and validation support')
    if not counts['train']['condiment_packet'] or any(
            r['name'].startswith('condiments__') and r['split'] != 'train' for r in records):
        raise ValueError('Keep the original condiment capture batch together in training')
    return counts


def build(source=SOURCE, output=DATA, reviews=REVIEWS, holds_file=None, external=None):
    source, output, reviews = Path(source).resolve(), Path(output).resolve(), Path(reviews).resolve()
    check_output(output, [source.parent])
    records, paths, moves = project(source, reviews, external)
    counts = audit(records)
    holds = dict(path=str(Path(holds_file).resolve()), sha256=digest(holds_file)) if holds_file else None
    training_holds(dict(records=records, training_holds=holds))
    for record in records:
        split, name = record['split'], record['name']
        image = output / 'images' / split / name
        label = output / 'labels' / split / Path(name).with_suffix('.txt')
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths[name], image)
        label.write_text('\n'.join(record['rows']) + ('\n' if record['rows'] else ''))
    (output / 'data.yaml').write_text(yaml.safe_dump(dict(path=str(output),
        train='images/train', val='images/val', names=class_names()), sort_keys=False))
    manifest = dict(source=str(source), source_fingerprint=dataset_fingerprint(source),
        external_reviews=(dict(path=str(Path(external).resolve()), sha256=digest(external)) if external else None),
        reviews=str(reviews), reviews_sha256=digest(reviews), records=records, counts=counts,
        moves=moves, training_holds=holds, image_counts=dict(Counter(r['split'] for r in records)),
        fingerprint=dataset_fingerprint(output / 'data.yaml'), independent_test=False,
        initialization='Fresh pretrained YOLO11n ancestry; checkpoints exposed to these held-out captures are excluded.',
        interpretation='Development validation; compare AP only with identical validation images, labels and inference settings. Source labels remain provisional outside targeted reviews.',
        annotation_review_scope='Hash-bound recovery-class additions and training-only target replacements; other corrected V2 rows preserved. Not a full 11-class semantic certification.',
        condiment_validation=(f"Limited: {counts['val']['condiment_packet']} annotated validation packet(s); small support does not establish packet accuracy."
            if counts['val']['condiment_packet'] else 'Unsupported: no packet validation annotations; the 44 original packet photos stay together as one background batch.'),
        grouping_limit='Recorded captures, original names, byte hashes and decoded pixels checked; unknown related captures may remain.')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def verify(output=DATA):
    output = Path(output).resolve()
    require_local([output / 'manifest.json'])
    manifest = json.loads((output / 'manifest.json').read_text())
    training_holds(manifest)
    if output == DATA and HOLDS.exists() and (manifest.get('training_holds') or {}).get('sha256') != digest(HOLDS):
        raise ValueError('Known training holds must be bound to this benchmark')
    source, reviews = Path(manifest['source']), Path(manifest['reviews'])
    require_local_dataset(source)
    require_local([reviews, *[p for folder in ('images', 'labels')
        for p in (output / folder).rglob('*') if p.is_file() and p.suffix.lower() in (*IMAGE_SUFFIXES, '.txt')]])
    if digest(reviews) != manifest['reviews_sha256'] or dataset_fingerprint(source) != manifest['source_fingerprint']:
        raise ValueError('Source/review provenance changed')
    external = manifest.get('external_reviews')
    if external:
        require_local([external['path']])
    if external and digest(Path(external['path'])) != external['sha256']:
        raise ValueError('External review provenance changed')
    records, _, moves = project(source, reviews, Path(external['path']) if external else None)
    if records != manifest['records'] or moves != manifest['moves']:
        raise ValueError('Source projection/provenance changed')
    expected_spec = dict(path=str(output), train='images/train', val='images/val', names=class_names())
    if yaml.safe_load((output / 'data.yaml').read_text()) != expected_spec:
        raise ValueError('Dataset YAML changed')
    for split in ('train', 'val'):
        expected = {r['name'] for r in records if r['split'] == split}
        actual = {p.name for p in (output / 'images' / split).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
        labels = {p.name for p in (output / 'labels' / split).glob('*.txt')}
        if actual != expected or labels != {str(Path(n).with_suffix('.txt')) for n in expected}:
            raise ValueError('Image/label membership changed')
    for record in records:
        image = output / 'images' / record['split'] / record['name']
        validate_image_file(image)
        if (digest(image) != record['sha'] or pixels(image) != record['pixels'] or
                label_path(image).read_text().splitlines() != record['rows']):
            raise ValueError('Image or labels changed: ' + record['name'])
    if (audit(records) != manifest['counts'] or
            dict(Counter(r['split'] for r in records)) != manifest['image_counts'] or
            dataset_fingerprint(output / 'data.yaml') != manifest['fingerprint']):
        raise ValueError('Dataset fingerprint/counts changed')
    print(json.dumps({key: manifest[key] for key in ('image_counts', 'counts', 'fingerprint', 'interpretation', 'condiment_validation')}, indent=2), flush=True)
    print('Integrity passed: original source files preserved; reviewed target corrections reproduced; no recorded capture/original/hash/pixel split overlap.', flush=True)
    return manifest


def training_holds(manifest):
    """Hold uncertain photos out of training without inventing negative labels."""
    evidence = manifest.get('training_holds')
    if not evidence:
        return set()
    path = Path(evidence['path'])
    if digest(path) != evidence['sha256']:
        raise ValueError('Training hold evidence changed')
    index = {r['name']: r for r in manifest['records']}
    seen = set()
    for hold in json.loads(path.read_text()):
        record = index.get(hold.get('name'))
        if (not record or record['split'] != 'train' or record['name'] in seen or
                hold.get('image_sha256') != record['sha'] or hold.get('rows') != record['rows'] or
                not hold.get('reviewer') or not hold.get('reason')):
            raise ValueError('Invalid or changed training hold')
        seen.add(record['name'])
    return seen


def verify_views(data, training_yaml, manifest):
    """A fast view list must retain every target positive and the exact holdout."""
    data, training_yaml = Path(data).resolve(), Path(training_yaml).resolve()
    require_local([training_yaml])
    spec = yaml.safe_load(training_yaml.read_text())
    root = Path(spec.get('path') or training_yaml.parent).resolve()
    val = Path(spec['val'])
    val = (val if val.is_absolute() else root / val).resolve()
    listing = Path(spec['train'])
    listing = listing if listing.is_absolute() else root / listing
    if (spec['names'] != class_names() or root != data or
            val != data / 'images/val' or not listing.is_file() or spec.get('test')):
        raise ValueError('Training views must retain names, source root and held-out validation')
    chosen = [Path(row).resolve() for row in listing.read_text().splitlines() if row.strip()]
    held = training_holds(manifest)
    allowed = {data / 'images/train' / r['name'] for r in manifest['records']
               if r['split'] == 'train' and r['name'] not in held}
    required = {data / 'images/train' / r['name'] for r in manifest['records']
                if r['split'] == 'train' and r['name'] not in held
                and any(int(row.split()[0]) in (3, 5, 7) for row in r['rows'])}
    if len(chosen) != len(set(chosen)) or not set(chosen) <= allowed:
        raise ValueError('Training list contains repeated, unknown or held-out training paths')
    if not required <= set(chosen):
        raise ValueError('Training list removed target-positive images')
    support = Counter(int(row.split()[0]) for r in manifest['records']
        if data / 'images/train' / r['name'] in set(chosen) for row in r['rows'])
    if any(not support[c] for c in (3, 5, 7)):
        raise ValueError('Training holds lost required target support')
    print(f'Training views verified: {len(chosen)} images; all {len(required)} eligible target-positive images retained; {len(held)} held photos excluded; validation unchanged.', flush=True)
    return chosen


def prepare_views(data, output):
    manifest = verify(data)
    config = prepare_weak_subset(data / 'data.yaml', output,
                                ['utensil', 'napkin', 'condiment_packet'], .15)
    held = training_holds(manifest)
    listing = config.parent / 'train.txt'
    rows = [row for row in listing.read_text().splitlines() if Path(row).name not in held]
    listing.write_text('\n'.join(rows) + '\n')
    details_path = output / 'weak_subset.json'
    details = json.loads(details_path.read_text())
    details['selected_training_views'] = len(rows)
    details['training_holds'] = sorted(held)
    selected = set(rows)
    support = Counter(class_names()[int(c)] for r in manifest['records']
        if str(data / 'images/train' / r['name']) in selected
        for c in {row.split()[0] for row in r['rows']})
    details['selected_class_group_support'] = {n: support[n] for n in class_names()}
    details['weak_original_groups'] = sum(any(int(row.split()[0]) in (3, 5, 7) for row in r['rows'])
        for r in manifest['records'] if str(data / 'images/train' / r['name']) in selected)
    details_path.write_text(json.dumps(details, indent=2) + '\n')
    verify_views(data, config, manifest)
    return config


def checkpoint_callback(output, counts, minutes=None, targets=('utensil',), stop_at_target=True):
    """Select by the weakest requested class; retain full optimizer state."""
    names = class_names()
    selection = output / 'utensil_selection.json'
    prior = json.loads(selection.read_text()) if selection.exists() else None
    best = min(prior['per_class_AP50'][c] for c in targets) if prior else -1.
    started = None

    def start(trainer):
        nonlocal started
        started = time.monotonic()

    def select(trainer):
        nonlocal best
        with Path(trainer.csv).open() as stream:
            rows = list(csv.DictReader(stream))
        if not rows or int(rows[-1]['epoch']) != trainer.epoch + 1:
            return
        metrics = trainer.validator.metrics
        indices = list(map(int, metrics.box.ap_class_index))
        if (list(map(int, metrics.nt_per_class)) != [counts[n] for n in names] or
                set(indices) != {i for i, n in enumerate(names) if counts[n]} or
                len(set(indices)) != len(indices)):
            raise ValueError('Validation class support changed')
        scores = {names[c]: float(metrics.box.ap50[j]) for j, c in enumerate(indices)}
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values()):
            raise ValueError('Invalid AP50 values')
        weights = output / 'train/weights'
        weights.mkdir(parents=True, exist_ok=True)
        atomic_checkpoint_copy(trainer.last, weights / 'last_resume.pt')
        score = min(scores[c] for c in targets)
        if score > best:
            best = score
            selected = weights / 'best_utensil.pt'
            atomic_checkpoint_copy(trainer.last, selected)
            selection.write_text(json.dumps(dict(epoch=trainer.epoch + 1, utensil_AP50=scores['utensil'],
                selection_targets=list(targets), selection_score=score,
                per_class_AP50=scores, weights=str(selected), weights_sha256=digest(selected)), indent=2) + '\n')
        print('Recovery AP50:', scores, flush=True)
        threshold = .85 if len(targets) > 1 else .9
        if (stop_at_target and score >= threshold) or (minutes is not None and time.monotonic() - started >= minutes * 60):
            trainer.stop = True

    select.start = start
    return select


def resume_options(checkpoint):
    # The coverage factory validates overrides before native check_resume loads
    # checkpoint arguments. State the required zeros at that boundary as well.
    return dict(resume=str(checkpoint), device='mps', workers=0,
                mosaic=0., mixup=0., cutmix=0., copy_paste=0.)


def validate_training_revision(previous, revised):
    """Only add training targets; keep every image, split and validation row fixed."""
    return validate_label_revision(previous, revised, 'train')


def validate_policy_extension(previous, revised):
    expected = {**previous, 'source_known_classes': {
        **previous['source_known_classes'], 'packet_extra': [7]}}
    if revised != expected:
        raise ValueError('Policy expansion may only add packet_extra coverage; old sources must remain unchanged')


def validate_expansion(previous, revised):
    """Keep all old images/holdouts; add training labels and new packet captures."""
    old = {r['name']: r for r in previous['records']}
    new = {r['name']: r for r in revised['records']}
    if (previous['source_fingerprint'] != revised['source_fingerprint'] or
            previous.get('training_holds') != revised.get('training_holds') or
            not set(old) <= set(new) or len(new) != len(revised['records'])):
        raise ValueError('Expansion changed original provenance, holds or image membership')
    added = Counter()
    for name, before in old.items():
        after = new[name]
        if {k: v for k, v in before.items() if k != 'rows'} != {k: v for k, v in after.items() if k != 'rows'}:
            raise ValueError('Expansion moved an old image or changed its provenance')
        if before['rows'] != after['rows']:
            if before['split'] != 'train' or after['rows'][:len(before['rows'])] != before['rows']:
                raise ValueError('Expansion changed an old validation label or deleted a training label')
            rows = validate_rows(after['rows'][len(before['rows']):], name, class_names())
            if any(int(row.split()[0]) not in (3, 5, 7) for row in rows):
                raise ValueError('Only recovery-class training additions allowed')
            added.update(class_names()[int(row.split()[0])] for row in rows)
    extra = [new[name] for name in set(new) - set(old)]
    if not extra or any(not r['name'].startswith('packet_extra__') or not r['rows']
                       or any(int(row.split()[0]) != 7 for row in r['rows']) for r in extra):
        raise ValueError('Expansion requires genuinely new reviewed packet images')
    for field in ('sha', 'pixels', 'group'):
        seen = {r[field] for r in old.values()}
        for record in extra:
            if record[field] in seen:
                raise ValueError('Expansion overlaps an old or duplicate original photo: ' + field)
            seen.add(record[field])
    if {r['capture'] for r in old.values()} & {r['capture'] for r in extra}:
        raise ValueError('Expanded packet captures overlap existing captures')
    audit(revised['records'])
    return dict(new_images=len(extra), added_training_objects=dict(added),
                existing_validation_images_and_labels_unchanged=True)


def validate_label_revision(previous, revised, split):
    """An annotation revision may add targets in exactly one existing split."""
    if split not in ('train', 'val'):
        raise ValueError('Expected a training or validation annotation revision')
    if (previous['source_fingerprint'] != revised['source_fingerprint'] or
            previous.get('training_holds') != revised.get('training_holds')):
        raise ValueError('Revision changed source provenance or training holds')
    old = {r['name']: r for r in previous['records']}
    new = {r['name']: r for r in revised['records']}
    if set(old) != set(new) or len(new) != len(revised['records']):
        raise ValueError('Revision changed image membership')
    added, changed = Counter(), 0
    for name, before in old.items():
        after = new[name]
        if {k: v for k, v in before.items() if k != 'rows'} != {k: v for k, v in after.items() if k != 'rows'}:
            raise ValueError('Revision changed image/split/provenance: ' + name)
        if after['rows'] == before['rows']:
            continue
        if before['split'] != split or after['rows'][:len(before['rows'])] != before['rows']:
            raise ValueError('Revision changed the other split or deleted existing labels: ' + name)
        additions = validate_rows(after['rows'][len(before['rows']):], name, class_names())
        if not additions or any(int(row.split()[0]) not in (3, 5, 7) for row in additions):
            raise ValueError('Revision requires additive recovery targets')
        changed += 1
        added.update(class_names()[int(row.split()[0])] for row in additions)
    if not changed:
        raise ValueError('Revision contains no new labels')
    return {('changed_training_images' if split == 'train' else 'changed_validation_images'): changed,
            'added_objects': dict(added), 'validation_unchanged': split == 'train'}


def revision_checkpoint(checkpoint, output, training_yaml, rebenchmark=False, nominal_batch=None):
    clone = continuation_checkpoint(checkpoint, output, Path(training_yaml).parent, 80)
    clone['train_args']['data'] = str(Path(training_yaml).resolve())
    if nominal_batch is not None:
        if nominal_batch not in (4, 64):
            raise ValueError('Nominal batch ablation must use 4 or 64')
        clone['train_args']['nbs'] = nominal_batch
    if rebenchmark:
        clone['best_fitness'] = 0.
        clone.pop('stopper', None)
    return clone


def prepare_revision(source, output, manifest, training_yaml, binding, rebenchmark=False, expand=False):
    """Fork an audited run without resetting weights, EMA, optimizer or schedule."""
    import torch

    source = Path(source).resolve()
    parent = source.parent.parent.parent
    if (source != parent / 'train/weights/last_resume.pt' or parent.parent != OUTPUT.parent or
            output.exists() or (parent / 'quality_hold.json').exists()):
        raise ValueError('Revision needs a clean recovery last_resume and fresh model directory')
    experiment = json.loads((parent / 'experiment.json').read_text())
    prior_binding = experiment['binding']
    prior_yaml = Path(prior_binding['training_yaml'])
    previous = verify(Path(yaml.safe_load(prior_yaml.read_text())['path']))
    verify_views(Path(yaml.safe_load(prior_yaml.read_text())['path']), prior_yaml, previous)
    prior_list = Path(yaml.safe_load(prior_yaml.read_text())['train'])
    if (previous['fingerprint'] != prior_binding['dataset_fingerprint'] or
            digest(prior_yaml) != prior_binding['training_yaml_sha256'] or
            digest(prior_list) != prior_binding['training_list_sha256'] or
            any(prior_binding[key] != binding[key] for key in
                ('pretrained_sha256', 'training_holds', 'epochs'))):
        raise ValueError('Parent experiment binding changed')
    prior_policy = Path(prior_binding.get('policy', POLICY))
    if expand:
        if digest(prior_policy) != prior_binding['policy_sha256']:
            raise ValueError('Parent policy changed')
        validate_policy_extension(json.loads(prior_policy.read_text()),
                                  json.loads(Path(binding['policy']).read_text()))
        revision = validate_expansion(previous, manifest)
        rebenchmark = True
    else:
        if prior_binding['policy_sha256'] != binding['policy_sha256']:
            raise ValueError('Parent policy changed')
        revision = validate_label_revision(previous, manifest, 'val' if rebenchmark else 'train')
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    validate_resume_epoch(source, checkpoint)
    if Path(checkpoint['train_args']['data']).resolve() != prior_yaml.resolve():
        raise ValueError('Parent checkpoint data mismatch')
    clone = revision_checkpoint(checkpoint, output, training_yaml, rebenchmark, binding.get('nominal_batch'))
    selected = json.loads((parent / 'utensil_selection.json').read_text())
    selected_path = Path(selected['weights'])
    if digest(selected_path) != selected['weights_sha256'] or selected['epoch'] > checkpoint['epoch'] + 1:
        raise ValueError('Parent selected checkpoint changed or is newer than continuation')
    baseline = None
    if rebenchmark:
        # Old AP cannot be inherited after labels change, even on the same images.
        from train import evaluate
        baseline = evaluate(selected_path, training_yaml, device='mps', imgsz=640, batch=4,
                            workers=0, name=output.name + '-inherited-rebenchmark')
        if digest(selected_path) != selected['weights_sha256']:
            raise ValueError('Checkpoint changed during rebenchmark evaluation')
        selected.update(utensil_AP50=baseline['per_class']['utensil']['ap50'],
            per_class_AP50={name: row['ap50'] for name, row in baseline['per_class'].items()
                           if row['ap50'] is not None})
        targets = binding.get('selection_targets', ['utensil'])
        selected.update(selection_targets=targets,
            selection_score=min(selected['per_class_AP50'][c] for c in targets))
    weights = output / 'train/weights'
    weights.mkdir(parents=True)
    torch.save(clone, weights / 'last_resume.pt')
    shutil.copy2(parent / 'train/results.csv', output / 'train/results.csv')
    atomic_checkpoint_copy(selected_path, weights / 'best_utensil.pt')
    selected.update(weights=str(weights / 'best_utensil.pt'), inherited_from=str(parent))
    (output / 'utensil_selection.json').write_text(json.dumps(selected, indent=2) + '\n')
    if baseline is not None:
        baseline.update(weights=str(weights / 'best_utensil.pt'),
            weights_sha256=selected['weights_sha256'], selected_epoch=selected['epoch'],
            dataset_fingerprint=manifest['fingerprint'], independent_test=False,
            validation_used_for_selection=True, interpretation=manifest['interpretation'])
        (output / 'inherited_rebenchmark.json').write_text(json.dumps(baseline, indent=2, allow_nan=False) + '\n')
    (output / 'experiment.json').write_text(json.dumps(dict(binding=binding,
        initialization='Continue existing clean optimizer/EMA; reviewed packet expansion.' if expand else 'Continue existing clean optimizer/EMA; additive annotations in one fixed split only.',
        source_checkpoint=str(source), source_checkpoint_sha256=digest(source),
        training_change=dict(nominal_batch_before=checkpoint['train_args'].get('nbs', 64),
            nominal_batch_after=clone['train_args'].get('nbs', 64),
            validation_batch=binding.get('validation_batch', 'native doubled training batch')),
        inherited_epochs=checkpoint['epoch'] + 1, revision=revision,
        checkpoint_selection='Highest minimum AP50 across bound selection targets on development validation; no independent test.'), indent=2) + '\n')
    print('Prepared training revision:', revision, flush=True)


class ReviewedCoverage:
    """Add verified image-level class coverage without changing the shared loss."""

    def __init__(self, base, reviewed):
        self.base, self.reviewed = base, reviewed
        self.names = base.names

    def validate_names(self, names):
        self.base.validate_names(names)

    def known_mask(self, im_files, batch_size, device):
        mask = self.base.known_mask(im_files, batch_size, device)
        for i, image in enumerate(im_files):
            mask[i, 0, list(self.reviewed.get(Path(image).name, ()))] = True
        return mask

    def to_dict(self):
        return {**self.base.to_dict(), 'reviewed_images': self.reviewed}


def reviewed_coverage(policy_path, manifest):
    """Only hash-bound, fully reviewed training classes may gain negative BCE."""
    evidence = json.loads(Path(policy_path).read_text()).get('image_reviews')
    if evidence is None:
        return {}
    path = Path(evidence['path'])
    require_local([path])
    if (digest(path) != evidence['sha256'] or
            manifest['fingerprint'] != evidence['dataset_fingerprint']):
        raise ValueError('Photo coverage evidence or dataset fingerprint changed')
    records = {r['name']: r for r in manifest['records']}
    result = {}
    for review in json.loads(path.read_text()):
        record = records.get(review.get('name'))
        classes = review.get('known_classes', [])
        absent = review.get('verified_absent', [])
        if (not record or record['split'] != 'train' or record['name'] in result or
                review.get('image_sha256') != record['sha'] or review.get('rows') != record['rows'] or
                not isinstance(classes, list) or not classes or len(set(classes)) != len(classes) or
                any(type(c) is not int or c not in (3, 7) for c in classes) or
                not isinstance(absent, list) or len(set(absent)) != len(absent) or
                set(absent) != {c for c in classes if not any(int(b.split()[0]) == c for b in record['rows'])} or
                not review.get('reviewer') or not review.get('reason')):
            raise ValueError('Photo coverage requires training identity, exact rows and verified class absence')
        result[record['name']] = tuple(classes)
    return result


def make_recovery_trainer(policy, validation_batch=None, reviewed=None, full_schedule=False):
    from partial_labels import make_trainer
    base = make_trainer(policy)
    if validation_batch is None and not reviewed and not full_schedule:
        return base

    class RecoveryTrainer(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if reviewed:
                self.coverage_policy = ReviewedCoverage(self.coverage_policy, reviewed)
                (self.save_dir / 'coverage_policy.json').write_text(
                    json.dumps(self.coverage_policy.to_dict(), indent=2) + '\n')

        def _setup_train(self):
            super()._setup_train()
            if full_schedule:
                self.args.patience = 0
                self.stopper.patience = math.inf

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
            return super().get_dataloader(dataset_path,
                validation_batch if mode == 'val' and validation_batch is not None else batch_size, rank, mode)

    return RecoveryTrainer


def train(args):
    import torch
    from ultralytics import YOLO
    from train import check_data, evaluate

    if not torch.backends.mps.is_available():
        raise ValueError('MPS is required for this experiment')
    if (args.rebenchmark or args.expand) and not args.continue_from:
        raise ValueError('--rebenchmark/--expand requires --continue-from and a fresh model directory')
    manifest = verify(args.data)
    verify_views(args.data, args.training_yaml, manifest)
    reviewed = reviewed_coverage(args.policy, manifest)
    print('Verified photo-level target coverage:', len(reviewed), 'training photos', flush=True)
    check_data(args.training_yaml)
    output = args.model_dir.resolve()
    if output.parent != OUTPUT.parent:
        raise ValueError('Use a separate child directory of model/utensil-focus')
    if (output / 'quality_hold.json').exists():
        raise ValueError('Model quality hold; start a clean experiment')
    pretrained = PROJECT_ROOT / 'model/weights/yolo11n.pt'
    spec = yaml.safe_load(args.training_yaml.read_text())
    binding = dict(dataset_fingerprint=manifest['fingerprint'],
        training_yaml=str(args.training_yaml.resolve()), training_yaml_sha256=digest(args.training_yaml),
        training_list_sha256=digest(Path(spec['train'])), policy_sha256=digest(args.policy),
        pretrained_sha256=digest(pretrained), training_holds=manifest.get('training_holds'), epochs=80)
    if args.policy.resolve() != POLICY.resolve():
        binding['policy'] = str(args.policy.resolve())
    if args.nominal_batch is not None:
        binding['nominal_batch'] = args.nominal_batch
    if args.validation_batch is not None:
        binding['validation_batch'] = args.validation_batch
    if args.full_schedule:
        binding['full_schedule'] = True
    if args.targets != ['utensil']:
        binding['selection_targets'] = args.targets
        for target in args.targets:
            positives = [r for r in manifest['records'] if r['split'] == 'val'
                and any(class_names()[int(row.split()[0])] == target for row in r['rows'])]
            if (manifest['counts']['val'][target] < 20 or len(positives) < 10
                    or len({r['capture'] for r in positives}) < 5):
                raise ValueError('Joint selection requires at least 20 boxes, 10 positive photos and 5 recorded capture groups per target; still no independent-test certification')
    if args.continue_from:
        if args.resume:
            raise ValueError('Choose --continue-from for a new revision or --resume for an existing run')
        prepare_revision(args.continue_from, output, manifest, args.training_yaml, binding, args.rebenchmark, args.expand)
        args.resume = True
        if args.prepare_only:
            print('Prepared optimizer-bearing continuation and fresh baseline evaluation; no training executed.', flush=True)
            return
    elif args.prepare_only:
        raise ValueError('--prepare-only requires --continue-from')
    if args.resume:
        experiment = json.loads((output / 'experiment.json').read_text())
        if experiment['binding'] != binding:
            raise ValueError('Resume dataset, training views, policy or epoch horizon changed')
        checkpoint_path = output / 'train/weights/last_resume.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        validate_resume_epoch(checkpoint_path, checkpoint)
        if (checkpoint.get('optimizer') is None or checkpoint['train_args']['epochs'] != 80 or
                Path(checkpoint['train_args']['data']).resolve() != args.training_yaml.resolve()):
            raise ValueError('Resume checkpoint lacks matching optimizer/data/80-epoch horizon')
        selected = json.loads((output / 'utensil_selection.json').read_text())
        if digest(Path(selected['weights'])) != selected['weights_sha256']:
            raise ValueError('Selected checkpoint hash changed')
        del checkpoint
        model = YOLO(str(checkpoint_path))
        options = resume_options(checkpoint_path)
    else:
        if output.exists():
            raise ValueError('Experiment exists; use --resume or a new --model-dir')
        model = YOLO(str(pretrained))
        if model.model.yaml.get('scale') != 'n':
            raise ValueError('Expected pretrained YOLO11n')
        options = dict(data=str(args.training_yaml.resolve()), epochs=80, batch=4, imgsz=640,
            device='mps', workers=0, seed=42, optimizer='AdamW', lr0=.001, warmup_epochs=3.,
            patience=0 if args.full_schedule else 15, project=str(output), name='train', exist_ok=False,
            mosaic=0., mixup=0., cutmix=0., copy_paste=0., cache=False, save=True)
        if args.nominal_batch is not None:
            options['nbs'] = args.nominal_batch
        output.mkdir(parents=True)
        (output / 'experiment.json').write_text(json.dumps(dict(binding=binding, training_args=options,
            initialization='Fresh pretrained YOLO11n; no previously trained held-out captures.',
            loss='Existing provisional coverage-aware loss; mixed-image augmentation disabled.',
            checkpoint_selection='Highest minimum AP50 across requested targets on development validation; independent test absent.'), indent=2) + '\n')
    callback = checkpoint_callback(output, manifest['counts']['val'], args.minutes, args.targets,
                                   stop_at_target=not args.full_schedule)
    model.add_callback('on_train_start', callback.start)
    model.add_callback('on_fit_epoch_end', callback)
    # Reconstruct the criterion factory explicitly on every resume.
    model.train(trainer=make_recovery_trainer(args.policy, args.validation_batch, reviewed,
                                            args.full_schedule), **options)
    # Release training/optimizer references before loading a second inference model on this 8GB MPS machine.
    import gc
    del model
    gc.collect()
    torch.mps.empty_cache()
    verify(args.data)
    verify_views(args.data, args.training_yaml, manifest)
    selected = json.loads((output / 'utensil_selection.json').read_text())
    weights = Path(selected['weights'])
    if digest(weights) != selected['weights_sha256']:
        raise ValueError('Selected checkpoint no longer matches its recorded score')
    result = evaluate(weights, args.training_yaml, device='mps', imgsz=640, batch=4, workers=0,
                      name=output.name + '-selected-eval')
    result.update(weights_sha256=digest(weights), dataset_fingerprint=manifest['fingerprint'],
        selected_epoch=selected['epoch'], independent_test=False,
        validation_used_for_selection=True, interpretation=manifest['interpretation'])
    (output / 'evaluation.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build', 'verify', 'views', 'train'))
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--reviews', type=Path, default=REVIEWS)
    parser.add_argument('--external', type=Path, help='Hash-bound, capture-grouped sauce-sachet photo reviews')
    parser.add_argument('--policy', type=Path, default=POLICY)
    parser.add_argument('--targets', nargs='+', choices=('utensil', 'condiment_packet'), default=['utensil'])
    parser.add_argument('--nominal-batch', type=int, choices=(4, 64), help='Explicit gradient-accumulation ablation; actual image batch stays 4')
    parser.add_argument('--validation-batch', type=int, choices=(4,), help='Avoid native validation batch doubling on MPS')
    parser.add_argument('--training-yaml', type=Path, help='Also verify an existing fast training list')
    parser.add_argument('--views-dir', type=Path, default=VIEWS.parent)
    parser.add_argument('--holds', type=Path, default=HOLDS)
    parser.add_argument('--model-dir', type=Path, default=OUTPUT)
    parser.add_argument('--minutes', type=float, help='Training budget, checked after complete epochs')
    parser.add_argument('--full-schedule', action='store_true', help='Complete 80 total epochs; disable early stopping and provisional AP target stopping')
    parser.add_argument('--resume', action='store_true', help='Resume the same 80-epoch horizon and optimizer')
    parser.add_argument('--continue-from', type=Path, help='Fork optimizer state after additive label corrections')
    parser.add_argument('--rebenchmark', action='store_true', help='Allow validation-only additions on the same images; freshly reevaluate inherited selection')
    parser.add_argument('--expand', action='store_true', help='Add reviewed packet captures and training labels; preserve old holdout and rebenchmark')
    parser.add_argument('--prepare-only', action='store_true', help='Verify/fork/rebenchmark without running a training epoch')
    args = parser.parse_args()
    if args.targets not in (['utensil'], ['utensil', 'condiment_packet']):
        parser.error('--targets must be utensil, or utensil condiment_packet in that order')
    args.data = args.data.resolve()
    if args.minutes is not None and (not math.isfinite(args.minutes) or args.minutes <= 0):
        parser.error('--minutes must be positive and finite')
    if args.command == 'train':
        args.training_yaml = args.training_yaml or VIEWS
        train(args)
        return
    if args.command == 'build':
        build(args.source, args.data, args.reviews, args.holds if args.holds.exists() else None, args.external)
    if args.command == 'views':
        prepare_views(args.data, args.views_dir.resolve())
        return
    manifest = verify(args.data)
    if args.training_yaml:
        verify_views(args.data, args.training_yaml, manifest)
    reviewed = reviewed_coverage(args.policy, manifest)
    if reviewed:
        print('Verified photo-level target coverage:', len(reviewed), 'training photos', flush=True)


if __name__ == '__main__':
    main()
