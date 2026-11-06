"""Audited, isolated six-class views of the corrected V5 archive."""
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import stat
import zipfile

from PIL import Image, ImageOps
import yaml

NAMES = ['food', 'plate', 'utensil', 'hinged_container', 'drink_carton', 'wrapper']
ARCHIVE_NAME = 'YOLO_V5_CommandCenter_B187_QC_selected_merged1572.zip'
PREFIX = 'v5_original_plus_v4/'
SPLITS = ('train', 'val', 'test')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def safe_path(name):
    p = PurePosixPath(name)
    if not name or '\\' in name or p.is_absolute() or '..' in p.parts or str(p) != name:
        raise ValueError(f'Unsafe or noncanonical path: {name}')
    return name


def strip_bottle(text):
    kept, removed = [], 0
    for line in text.splitlines():
        if not line.strip():
            continue
        p = line.split()
        if len(p) != 5 or p[0] not in {str(i) for i in range(7)}:
            raise ValueError('Invalid YOLO class or row')
        x, y, w, h = map(float, p[1:])
        if not (all(math.isfinite(v) for v in (x, y, w, h)) and 0 < w <= 1 and 0 < h <= 1
                and -1e-5 <= x-w/2 <= x+w/2 <= 1+1e-5
                and -1e-5 <= y-h/2 <= y+h/2 <= 1+1e-5):
            raise ValueError('Invalid YOLO geometry')
        if p[0] == '6':
            removed += 1
        else:
            kept.append(' '.join(p))
    return '\n'.join(kept) + ('\n' if kept else ''), removed


def read_archive(path, expected_images=2261):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Corrected V5 archive is unavailable: {path}')
    contents, removed, counts = {}, Counter(), {s: Counter() for s in SPLITS}
    with zipfile.ZipFile(path) as z:
        names = set()
        for entry in z.infolist():
            name = entry.filename.rstrip('/') if entry.is_dir() else entry.filename
            safe_path(name)
            if not name.startswith(PREFIX) and name != PREFIX.rstrip('/'):
                raise ValueError(f'Unexpected archive path: {name}')
            if entry.filename in names:
                raise ValueError(f'Duplicate ZIP member: {entry.filename}')
            names.add(entry.filename)
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError('Archive symlink is not allowed')
        rows = json.loads(z.read(PREFIX + 'manifest.json'))
        if not isinstance(rows, list) or len(rows) != expected_images:
            raise ValueError(f'Expected {expected_images} manifest images')
        cfg = yaml.safe_load(z.read(PREFIX + 'data.yaml'))
        names_cfg = cfg['names']
        if isinstance(names_cfg, dict):
            names_cfg = [names_cfg.get(i) for i in range(7)]
        if names_cfg != NAMES + ['bottle']:
            raise ValueError('Wrong original seven-class mapping')
        split_counts = Counter(r['split'] for r in rows)
        if set(split_counts) != set(SPLITS):
            raise ValueError('Expected train/val/test splits')
        if expected_images == 2261 and split_counts != dict(train=2093, val=97, test=71):
            raise ValueError('Unexpected original V5 split counts')
        seen_paths = set()
        for r in rows:
            for kind in ('image', 'label'):
                rel = safe_path(r[kind + '_path'])
                expected_prefix = ('images/' if kind == 'image' else 'labels/') + r['split'] + '/'
                if not rel.startswith(expected_prefix):
                    raise ValueError('Manifest path and split disagree')
                if rel in seen_paths:
                    raise ValueError('Duplicate manifest path')
                seen_paths.add(rel)
                data = z.read(PREFIX + rel)
                if digest(data) != r[kind + '_sha256']:
                    raise ValueError(f'Source checksum mismatch: {rel}')
                contents[rel] = data
            ip, lp = PurePosixPath(r['image_path']), PurePosixPath(r['label_path'])
            if ip.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'} or lp.suffix != '.txt' or ip.stem != lp.stem:
                raise ValueError('Image/label pair has incompatible names or format')
            r['label_text'], n = strip_bottle(contents[r['label_path']].decode('utf-8-sig'))
            r['counts'] = dict(Counter(int(line.split()[0]) for line in r['label_text'].splitlines()))
            removed[r['split']] += n
            counts[r['split']].update(r['counts'])
        # Verify CRC of auxiliary QC ledgers as well as manifest data.
        if z.testzip() is not None:
            raise ValueError('Archive CRC failure')
    source = dict(archive=str(path), archive_sha256=sha(path), images=len(rows),
                  original_splits=dict(split_counts), removed_bottle_boxes=dict(removed),
                  retained_class_counts={s: {NAMES[c]: counts[s][c] for c in range(6)} for s in SPLITS},
                  annotation_status='provisional; not independently human verified')
    return rows, contents, source


def validate_exclusions(exclusions, rows):
    paths = {r['image_path'] for r in rows}
    if len(exclusions) != 28 or len(set(exclusions)) != 28 or not set(exclusions) <= paths:
        raise ValueError('Exclusions must list exactly 28 unique, exact manifest image_path values')
    return set(exclusions)


def load_exclusions(path):
    path = Path(path)
    if path.suffix.lower() == '.json':
        items = json.loads(path.read_text())
        if not isinstance(items, list) or any(not isinstance(p, str) for p in items):
            raise ValueError('Exclusion JSON must be a list of image_path strings')
        return items
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or 'image_path' not in reader.fieldnames:
            raise ValueError('Exclusion CSV needs an image_path column')
        return [r['image_path'] for r in reader]


def image_features(rows, contents):
    import imagehash
    result = []
    for r in rows:
        with Image.open(io.BytesIO(contents[r['image_path']])) as image:
            im = ImageOps.exif_transpose(image).convert('RGB')
            result.append(dict(phash=int(str(imagehash.phash(im)), 16),
                               dhash=int(str(imagehash.dhash(im)), 16),
                               aspect=im.width / im.height,
                               pixels=digest(str(im.size).encode() + im.tobytes())))
    return result


def assign_splits(rows, features):
    n = len(rows)
    if n < 20 or len(features) != n:
        raise ValueError('Insufficient cohort for supported group-aware split')
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = root(a), root(b)
        parent[max(a, b)] = min(a, b)

    seen = {}
    for i, (r, f) in enumerate(zip(rows, features)):
        keys = [('bytes', r['image_sha256'])]
        if f.get('pixels'):
            keys.append(('pixels', f['pixels']))
        for key in ('meal_id', 'tray_pair_id', 'pair_id', 'group_id', 'session_id'):
            if r.get(key) is not None and str(r[key]).strip():
                keys.append((key, r.get('dataset', r.get('source_dataset', '')), str(r[key])))
        for key in keys:
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
    for i in range(n):
        for j in range(i):
            a, b = features[i], features[j]
            if (abs(a['aspect']-b['aspect']) / max(a['aspect'], b['aspect']) <= .025
                    and (a['phash'] ^ b['phash']).bit_count() <= 3
                    and (a['dhash'] ^ b['dhash']).bit_count() <= 4):
                union(i, j)
    grouped = defaultdict(list)
    for i in range(n):
        grouped[root(i)].append(i)
    groups = {}
    selected = {}
    size = Counter()
    boxes = {s: Counter() for s in SPLITS}
    for gid, inds in grouped.items():
        counts = sum((Counter(rows[i]['counts']) for i in inds), Counter())
        groups[gid] = dict(members=inds, size=len(inds), counts=counts)
        held = {rows[i]['split'] for i in inds}
        fixed = 'test' if 'test' in held else 'val' if 'val' in held else None
        if fixed:
            selected[gid] = fixed
            size[fixed] += len(inds)
            boxes[fixed].update(counts)
    target = round(n * .1)
    if any(size[s] > .12 * n for s in ('val', 'test')):
        raise ValueError('Held-out duplicate groups prevent safe 80/10/10 split')
    remaining = set(groups) - set(selected)

    def take(gid, split):
        selected[gid] = split
        remaining.remove(gid)
        size[split] += groups[gid]['size']
        boxes[split].update(groups[gid]['counts'])

    frequency = {c: sum(g['counts'][c] > 0 for g in groups.values()) for c in range(6)}
    for c in sorted(range(6), key=lambda c: (frequency[c], c)):
        minimum = 2 if frequency[c] >= 6 else 1
        if frequency[c] < 3:
            raise ValueError(f'Insufficient independent group support: {NAMES[c]}')
        for s in ('val', 'test'):
            while sum(groups[g]['counts'][c] > 0 for g in selected if selected[g] == s) < minimum:
                options = [g for g in remaining if groups[g]['counts'][c] > 0
                           and size[s] + groups[g]['size'] <= target]
                if not options:
                    raise ValueError(f'Cannot allocate independent class support for {NAMES[c]} in {s}')
                take(min(options, key=lambda g: (groups[g]['size'], -min(groups[g]['counts'][c], 5), g)), s)
    total = sum((g['counts'] for g in groups.values()), Counter())
    for s in ('val', 'test'):
        def score(g):
            return (sum(((boxes[s][c]+groups[g]['counts'][c]-total[c]*.1)**2
                         - (boxes[s][c]-total[c]*.1)**2) / max(1, total[c]*.1)
                        for c in range(6)) / groups[g]['size'], groups[g]['size'], g)
        while size[s] < target:
            options = [g for g in remaining if groups[g]['size'] <= target-size[s]]
            if not options:
                break
            take(min(options, key=score), s)
    for g in remaining:
        selected[g] = 'train'
        size['train'] += groups[g]['size']
        boxes['train'].update(groups[g]['counts'])
    if any(not .08*n <= size[s] <= .12*n for s in ('val', 'test')):
        raise ValueError('Group sizes prevent safe 80/10/10 split')
    if any(boxes[s][c] == 0 for s in SPLITS for c in range(6)):
        raise ValueError('Missing class support after splitting')
    return [selected[root(i)] for i in range(n)], [root(i) for i in range(n)]


def materialize(rows, contents, source, splits, groups, output, exclusions):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite dataset: {output}')
    output.mkdir(parents=True)
    records, files = [], {}
    counts = {s: Counter() for s in SPLITS}
    for i, (row, split, group) in enumerate(zip(rows, splits, groups)):
        stem = f'{i:05d}'
        ip = f'images/{split}/{stem}{PurePosixPath(row["image_path"]).suffix.lower()}'
        lp = f'labels/{split}/{stem}.txt'
        for rel, data in ((ip, contents[row['image_path']]), (lp, row['label_text'].encode())):
            p = output / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            files[rel] = digest(data)
        records.append(dict(image_path=ip, label_path=lp, image_sha256=files[ip],
                            label_sha256=files[lp], split=split, group=group,
                            original_split=row['split'], original_image=row['image_path'],
                            source_label_sha256=row['label_sha256'], counts=row['counts']))
        counts[split].update(row['counts'])
    for split in SPLITS:
        rel = split + '.txt'
        (output / rel).write_text(''.join(str(output / r['image_path']) + '\n'
                                        for r in records if r['split'] == split))
        files[rel] = sha(output / rel)
    (output / 'data.yaml').write_text(yaml.safe_dump(dict(
        path=str(output), train='train.txt', val='val.txt', test='test.txt', nc=6, names=NAMES), sort_keys=False))
    files['data.yaml'] = sha(output / 'data.yaml')
    receipt = dict(schema=1, names=NAMES, source=source, files=files, images=records,
                   exclusions=sorted(exclusions), split_counts=dict(Counter(splits)),
                   class_counts={s: {NAMES[c]: counts[s][c] for c in range(6)} for s in SPLITS},
                   annotation_status='provisional; not independently human verified')
    write_json(output / 'dataset_receipt.json', receipt)
    return receipt


def prepare(archive, output, exclusions):
    if Path(archive).name != ARCHIVE_NAME:
        raise ValueError(f'Use the corrected archive named {ARCHIVE_NAME}; no older-data fallback')
    if Path(output).exists():
        raise FileExistsError(output)
    rows, contents, source = read_archive(archive)
    excluded = validate_exclusions(load_exclusions(exclusions), rows)
    source['exclusion_ledger_sha256'] = sha(exclusions)
    rows = [r for r in rows if r['image_path'] not in excluded]
    features = image_features(rows, contents)
    splits, groups = assign_splits(rows, features)
    return materialize(rows, contents, source, splits, groups, output, excluded)


def verify_dataset(output, require_support=True):
    output = Path(output).resolve()
    receipt = json.loads((output / 'dataset_receipt.json').read_text())
    if receipt['names'] != NAMES:
        raise ValueError('Changed six-class names')
    for rel, expected in receipt['files'].items():
        safe_path(rel)
        p = output / rel
        if not p.is_file() or p.is_symlink() or sha(p) != expected:
            raise ValueError(f'Changed dataset file: {rel}')
    group_splits, image_splits, seen = {}, {}, set()
    actual = {s: Counter() for s in SPLITS}
    for r in receipt['images']:
        s = r['split']
        if s not in SPLITS or r['image_path'] in seen:
            raise ValueError('Duplicate image or invalid split')
        seen.add(r['image_path'])
        for kind in ('image', 'label'):
            rel = safe_path(r[kind + '_path'])
            if receipt['files'].get(rel) != r[kind + '_sha256']:
                raise ValueError('Receipt image/label hashes disagree')
        text, removed = strip_bottle((output / r['label_path']).read_text())
        if removed:
            raise ValueError('Bottle remains in six-class dataset')
        counts = Counter(int(line.split()[0]) for line in text.splitlines())
        if dict(counts) != {int(c): n for c, n in r['counts'].items()}:
            raise ValueError('Receipt class counts disagree')
        actual[s].update(counts)
        if r['original_split'] != 'train' and s == 'train':
            raise ValueError('Historical held-out image entered training')
        for mapping, key in ((group_splits, r['group']), (image_splits, r['image_sha256'])):
            if key in mapping and mapping[key] != s:
                raise ValueError('Group/image overlap between splits')
            mapping[key] = s
    cfg = yaml.safe_load((output / 'data.yaml').read_text())
    if cfg != dict(path=str(output), train='train.txt', val='val.txt', test='test.txt', nc=6, names=NAMES):
        raise ValueError('Changed dataset configuration; dataset cannot be moved without rebuilding lists')
    for s in SPLITS:
        expected = [str(output / r['image_path']) for r in receipt['images'] if r['split'] == s]
        if (output / (s + '.txt')).read_text().splitlines() != expected:
            raise ValueError('Image list differs from receipt')
        if require_support and any(actual[s][c] == 0 for c in range(6)):
            raise ValueError(f'Missing class support in {s}')
        if receipt['class_counts'][s] != {NAMES[c]: actual[s][c] for c in range(6)}:
            raise ValueError('Changed class support totals')
    return receipt
