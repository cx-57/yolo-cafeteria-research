"""Reviewed two-class YOLO11n specialist; frozen development validation.

build -> initialize -> evaluate -> train. All outputs are isolated from V2.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
import random
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/utensil-focus/grouped-v7-reviewed-negatives'
DATA = ROOT / 'data/utensil-packet-focus/reviewed-v1'
RUN = ROOT / 'model/utensil-packet-focus/warm-low-v1'
WEIGHTS = ROOT / 'model/utensil-focus/recovery80-joint-v6/train/weights/best_utensil.pt'
NAMES = ['utensil', 'condiment_packet']
BALANCED = False
EXPANDED = False
CONTEXT = False
COCO_RESTART = False
HEAD_ONLY = False
REAL_ONLY = False
BALANCED_REAL = False
MEAL_SCENES = False
SOURCE_FP = 'cc30ea3b8886e87caf6252f9ee62dd5cdd3814bbf774072a4d7beb224aa3a1bd'
WEIGHTS_SHA = '5cf0eca1767d703b1172ffaf616e8f8d9d43c98e93863946bcd61f122e516f48'


def read_bytes(path):
    path = Path(path)
    if getattr(path.stat(), 'st_flags', 0) & 0x40000000:
        raise ValueError(f'iCloud file is not downloaded: {path}')
    raw = path.read_bytes()
    if len(raw) != path.stat().st_size:
        raise ValueError(f'Incomplete file: {path}')
    return raw


def sha(path):
    return hashlib.sha256(read_bytes(path)).hexdigest()


def load(path):
    return json.loads(read_bytes(path))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def project_records(records, reviewed, held):
    result = []
    for record in records:
        train = record['split'] == 'train'
        if train and (record['name'] in held or not {3, 7} <= set(reviewed.get(record['name'], ()))):
            continue
        item = copy.deepcopy(record)
        item['rows'] = [str({3: 0, 7: 1}[int(row.split()[0])]) + ' ' + row.split(' ', 1)[1]
                        for row in record['rows'] if int(row.split()[0]) in (3, 7)]
        item['verified_absence'] = train and not item['rows']
        result.append(item)
    return result


def coco_restart_model(source):
    """Reuse COCO features/localization, with a fresh target-specific classifier."""
    from ultralytics.nn.tasks import DetectionModel
    if source.model[-1].nc != 80:
        raise ValueError('Expected the original 80-class COCO pretrained model')
    model = DetectionModel(copy.deepcopy(source.yaml), nc=2, verbose=False).float()
    model.load(source)
    model.names = dict(enumerate(NAMES))
    return model


def project_model(model):
    """Keep learned boxes and exact class-3/7 logits; never reuse an 11-class EMA."""
    import torch
    model = copy.deepcopy(model).float()
    head = model.model[-1]
    if head.nc != 11 or head.end2end:
        raise ValueError('Expected an ordinary 11-class detection head')
    for branch in head.cv3:
        old = branch[-1]
        new = torch.nn.Conv2d(old.in_channels, 2, old.kernel_size, old.stride,
                              old.padding, bias=old.bias is not None).to(old.weight)
        with torch.no_grad():
            new.weight.copy_(old.weight[[3, 7]])
            if old.bias is not None:
                new.bias.copy_(old.bias[[3, 7]])
        branch[-1] = new
    head.nc = 2
    head.no = 4 * head.reg_max + 2
    model.yaml['nc'] = 2
    model.names = dict(enumerate(NAMES))
    model.criterion = None
    return model


def repeat_packet_records(records, source_root, copies=4):
    """Weight real training positives; preserve their original split identities."""
    if copies < 1:
        raise ValueError('copies must be positive')
    result = copy.deepcopy(records)
    for original in records:
        if original['split'] != 'train' or not any(row.startswith('1 ') for row in original['rows']):
            continue
        for repeat in range(1, copies):
            record = copy.deepcopy(original)
            record['name'] = f'repeat{repeat}__{original["name"]}'
            record['image_path'] = str(original.get('image_path', source_root/'images/train'/original['name']))
            record['parents'] = [dict(name=original['name'], sha=original['sha'], split='train')]
            record['training_repetition'] = True
            result.append(record)
    if len({r['name'] for r in result}) != len(result):
        raise ValueError('Repeated sample name collision')
    return result


def balance_packets(records):
    """Four visually chosen red/orange front/back views; cap one repeated capture."""
    keep = ('464122_', '464135_', '464147_', '464164_')
    legacy = [r for r in records if r['split']=='train' and r['name'].startswith('condiments__')]
    selected = [r for r in legacy if r['name'].split('__')[-1].startswith(keep)]
    if len(legacy)!=44 or len(selected)!=4:
        raise ValueError('Unexpected legacy packet membership')
    names = {r['name'] for r in selected}
    return [r for r in records if r not in legacy or r['name'] in names]


def render_composite(background, objects, output, seed):
    """Training augmentation using reviewed source pixels and verified empty scenes."""
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
    objects = [objects] if isinstance(objects, dict) else objects
    if len(objects) not in (1,2):
        raise ValueError('Expected one or two packet cutouts')
    if (background['split']!='train' or background['rows'] or not background.get('verified_absence')
            or set(background.get('known_classes', []))!={0,1}):
        raise ValueError('Composite background must be reviewed, joint-negative TRAIN data')
    for obj in objects:
        if obj['split']!='train' or set(obj.get('known_classes', []))!={0,1} or not obj.get('polygon'):
            raise ValueError('Composite packet must be a reviewed TRAIN source with a mask')
    rng = random.Random(seed)
    if sha(background['image_path'])!=background['sha']:
        raise ValueError('Background source changed')
    image = Image.open(background['image_path']).convert('RGB')
    image.thumbnail((640,640))
    w,h = image.size
    image = image.convert('RGBA')
    rows, boxes = [], []
    horizontal = rng.random()<.5
    for object_index,obj in enumerate(objects):
        if sha(obj['image_path'])!=obj['sha']:
            raise ValueError('Packet source changed')
        packet = Image.open(obj['image_path']).convert('RGB')
        mask = Image.new('L',packet.size,0)
        ImageDraw.Draw(mask).polygon([tuple(p) for p in obj['polygon']],fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(.4))
        packet = ImageEnhance.Brightness(packet).enhance(rng.uniform(.85,1.1)).convert('RGBA')
        packet.putalpha(mask)
        packet = packet.crop(packet.getbbox())
        # Include small sachets on a tray and larger close-up objects.
        area = math.exp(rng.uniform(math.log(.008), math.log(.28)))*w*h
        scale = math.sqrt(area/(packet.width*packet.height))
        packet = packet.resize((max(8,round(packet.width*scale)),max(8,round(packet.height*scale))),Image.Resampling.LANCZOS)
        packet = packet.rotate(rng.uniform(-180,180),resample=Image.Resampling.BICUBIC,expand=True)
        packet = packet.crop(packet.getbbox())
        max_w = .45*w if len(objects)>1 and horizontal else .8*w
        max_h = .45*h if len(objects)>1 and not horizontal else .8*h
        fit = min(1,max_w/packet.width,max_h/packet.height)
        if fit<1:
            packet = packet.resize((max(1,round(packet.width*fit)),max(1,round(packet.height*fit))),Image.Resampling.LANCZOS)
            packet = packet.crop(packet.getbbox())
        pw,ph = packet.size
        for attempt in range(100):
            if len(objects)>1 and horizontal:
                offset = object_index*(w//2)
                left,top = rng.randint(offset,offset+w//2-pw-2),rng.randint(0,h-ph)
            elif len(objects)>1:
                offset = object_index*(h//2)
                left,top = rng.randint(0,w-pw),rng.randint(offset,offset+h//2-ph-2)
            else:
                left,top = rng.randint(0,w-pw),rng.randint(0,h-ph)
            box = (left,top,left+pw,top+ph)
            if not any(min(box[2],b[2])>max(box[0],b[0])-2 and min(box[3],b[3])>max(box[1],b[1])-2 for b in boxes):
                break
        else:
            raise ValueError('Cannot place non-overlapping packet cutouts')
        image.alpha_composite(packet,(left,top))
        boxes.append(box)
        rows.append(f'1 {(left+pw/2)/w:.9f} {(top+ph/2)/h:.9f} {pw/w:.9f} {ph/h:.9f}')
    image.convert('RGB').save(output)
    return dict(name=Path(output).name,split='train',sha=sha(output),
                pixels=hashlib.sha256(str((w,h)).encode()+image.convert('RGB').tobytes()).hexdigest(),
                group=Path(output).stem,capture=None,rows=rows,verified_absence=False,
                synthetic=True,seed=seed,parents=[dict(name=r['name'],sha=r['sha'],split=r['split']) for r in [background,*objects]])


def apply_training_corrections(records, reviews):
    records = copy.deepcopy(records)
    by_name = {r['name']:r for r in records}
    for review in reviews:
        r = by_name.get(review['name'])
        if (not r or r['split']!='train' or r['sha']!=review['sha'] or r['rows']!=review['expected_rows']
                or not review['reviewer'] or not review['reason']):
            raise ValueError('Training correction is unbound or crosses validation')
        r['rows'] = list(review['rows'])
        r['verified_absence'] = not r['rows']
        r['known_classes'] = [0,1]
        r['training_correction'] = dict(reviewer=review['reviewer'],reason=review['reason'])
    return records


def counts(records):
    result = {}
    for split in ('train', 'val'):
        items = [r for r in records if r['split'] == split]
        objects = Counter(int(row.split()[0]) for r in items for row in r['rows'])
        result[split] = dict(images=len(items), objects={n: objects[i] for i, n in enumerate(NAMES)},
                             verified_joint_negatives=sum(r['verified_absence'] for r in items))
    return result


def validate_configuration(root, records):
    import yaml
    config = yaml.safe_load(read_bytes(root/'data.yaml'))
    expected = dict(path=str(root), train='images/train', val='images/val', names=NAMES)
    if config != expected:
        raise ValueError('Dataset YAML differs from the reviewed configuration')
    for split in ('train', 'val'):
        if any(p.is_dir() for folder in ('images','labels') for p in (root/folder/split).iterdir()):
            raise ValueError(f'Unexpected nested dataset directory: {split}')
        names = {r['name'] for r in records if r['split'] == split}
        images = {p.name for p in (root/'images'/split).iterdir() if p.is_file()}
        labels = {p.name for p in (root/'labels'/split).iterdir() if p.is_file()}
        if images != names or labels != {str(Path(n).with_suffix('.txt')) for n in names}:
            raise ValueError(f'Unreviewed or missing dataset files: {split}')


def validate(records, root):
    """Check current bytes, geometry, labels and all recorded leakage identities."""
    from PIL import Image
    identities = {}
    by_name = {r['name']:r for r in records}
    for r in records:
        for parent in r.get('parents', []):
            actual_parent = by_name.get(parent['name'])
            if (r['split']!='train' or not actual_parent or actual_parent['split']!='train'
                    or parent['sha']!=actual_parent['sha'] or parent['split']!='train'):
                raise ValueError('Synthetic source ancestry is invalid or crosses validation')
        image = root / 'images' / r['split'] / r['name']
        if sha(image) != r['sha']:
            raise ValueError(f'Image changed: {image}')
        with Image.open(image) as im:
            rgb = im.convert('RGB')
            actual_pixels = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
        label = root / 'labels' / r['split'] / Path(r['name']).with_suffix('.txt')
        if read_bytes(label).decode().splitlines() != r['rows']:
            raise ValueError(f'Label changed: {label}')
        for row in r['rows']:
            c, x, y, w, h = map(float, row.split())
            if (c not in (0, 1) or not all(math.isfinite(v) for v in (x, y, w, h))
                    or min(w, h) <= 0 or min(x-w/2, y-h/2) < -1e-6
                    or max(x+w/2, y+h/2) > 1+1e-6):
                raise ValueError(f'Invalid box: {r["name"]}: {row}')
        for key in ('name', 'sha', 'pixels', 'group', 'capture'):
            identity = (key, r.get(key))
            if identity[1] is None:
                continue
            if identity in identities and identities[identity] != r['split']:
                raise ValueError(f'Split leakage: {identity}')
            identities[identity] = r['split']
        identity = ('decoded_pixels', actual_pixels)
        if identity in identities and identities[identity] != r['split']:
            raise ValueError(f'Decoded-pixel split leakage: {r["name"]}')
        identities[identity] = r['split']


def build():
    import yaml
    if DATA.exists():
        raise ValueError(f'Output exists: {DATA}; use verify to inspect it')
    source = load(SOURCE / 'manifest.json')
    assert source['fingerprint'] == SOURCE_FP
    policy_path = ROOT / 'data/utensil-focus/coverage-reviewed-v7.json'
    policy = load(policy_path)
    proof = policy['image_reviews']
    assert proof['dataset_fingerprint'] == SOURCE_FP and sha(proof['path']) == proof['sha256']
    by_name = {r['name']: r for r in source['records']}
    reviewed = {}
    for review in load(proof['path']):
        original = by_name[review['name']]
        assert original['split'] == 'train'
        assert original['sha'] == review['image_sha256'] and original['rows'] == review['rows']
        assert review['reviewer'] and review['reason']
        present = {int(row.split()[0]) for row in original['rows']}
        assert set(review['known_classes']) - present == set(review['verified_absent'])
        reviewed[review['name']] = review['known_classes']
    assert sha(source['training_holds']['path']) == source['training_holds']['sha256']
    held = {r['name'] for r in load(source['training_holds']['path'])}
    records = project_records(source['records'], reviewed, held)
    if BALANCED:
        records = balance_packets(records)
    extra_path = ROOT/'data/utensil-packet-focus/extra-reviews-v3.json'
    if EXPANDED:
        import html
        import re
        def author(s): return html.unescape(re.sub('<[^>]+>','',s)).strip().casefold()
        val_authors = {author(r['artist']) for r in load(source['external_reviews']['path']) if r['split']=='val'}
        for r in load(extra_path):
            assert r['split']=='train' and set(r['known_classes'])=={0,1}
            assert r['rows'] and r['reviewer'] and r['reason'] and not r['verified_absence']
            assert author(r['provenance']['artist']) not in val_authors
            assert sha(r['image_path'])==r['sha']
            records.append(r)
    negative_path = ROOT/'data/utensil-packet-focus/negative-reviews-v4.json'
    polygon_path = ROOT/'data/utensil-packet-focus/packet-polygons-v4.json'
    correction_path = ROOT/'data/utensil-packet-focus/training-corrections-v4.json'
    if CONTEXT:
        for r in load(negative_path):
            original = by_name[r['name']]
            assert original['split']=='train' and original['sha']==r['sha'] and original['rows']==r['original_rows']
            assert r['rows']==[] and r['verified_absence'] and set(r['known_classes'])=={0,1}
            assert r['reviewer'] and r['reason']
            records.append(r)
        records = apply_training_corrections(records,load(correction_path))
    assert counts(records)['train']['images'] == (227 if CONTEXT else 192 if EXPANDED else 184 if BALANCED else 224)
    assert counts(records)['val']['images'] == 764
    if BALANCED_REAL:
        records = repeat_packet_records(records, SOURCE)
    meal_path = ROOT/'data/utensil-packet-focus/meal-scene-reviews-v10.json'
    if MEAL_SCENES:
        for r in load(meal_path):
            assert r['split']=='train' and set(r['known_classes'])=={0,1}
            assert r['reviewer'] and r['reason'] and r['rows']
            assert author(r['provenance']['artist']) not in val_authors
            assert sha(r['image_path'])==r['sha']
            assert all(r['sha']!=old['sha'] and r['pixels']!=old['pixels'] for old in source['records'])
            records.append(r)
    for r in records:
        dest = DATA / 'images' / r['split'] / r['name']
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(r.get('image_path', SOURCE / 'images' / r['split'] / r['name']))
        label = DATA / 'labels' / r['split'] / Path(r['name']).with_suffix('.txt')
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text('\n'.join(r['rows']) + ('\n' if r['rows'] else ''))
    if CONTEXT and not REAL_ONLY:
        backgrounds = []
        for r in records:
            if r['split']=='train' and r['verified_absence']:
                backgrounds.append(dict(r,known_classes=[0,1],image_path=str(DATA/'images/train'/r['name'])))
        objects = load(polygon_path)
        for obj in objects:
            assert obj['name'] in {r['name'] for r in records if r['split']=='train'}
            assert obj['sha']==next(r['sha'] for r in records if r['name']==obj['name'])
        rng = random.Random(42)
        for i in range(128):
            background = backgrounds[i%len(backgrounds)]
            selected = rng.sample(objects,1 if i%2==0 else 2)
            image_path = DATA/'images/train'/f'packet_context_v4_{i:03d}.png'
            r = render_composite(background,selected,image_path,seed=42000+i)
            (DATA/'labels/train'/Path(r['name']).with_suffix('.txt')).write_text('\n'.join(r['rows'])+'\n')
            records.append(r)
    validate(records, DATA)
    manifest = dict(records=records, counts=counts(records), source_fingerprint=SOURCE_FP,
                    real_packet_sampling_copies=4 if BALANCED_REAL else 1,
                    meal_scene_reviews_sha=sha(meal_path) if MEAL_SCENES else None,
                    legacy_packet_cap=4 if BALANCED else None,
                    source_manifest_sha=sha(SOURCE / 'manifest.json'), coverage_sha=sha(policy_path),
                    review_sha=proof['sha256'], independent_test=False,
                    extra_reviews_sha=sha(extra_path) if EXPANDED else None,
                    negative_reviews_sha=sha(negative_path) if CONTEXT else None,
                    polygon_reviews_sha=sha(polygon_path) if CONTEXT else None,
                    training_corrections_sha=sha(correction_path) if CONTEXT else None,
                    interpretation='Fixed development validation; inherited validation completeness is provisional. '
                    'Only training images fully reviewed for both classes. No new independent cafeteria session.')
    manifest['fingerprint'] = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    save(DATA / 'manifest.json', manifest)
    (DATA / 'data.yaml').write_text(yaml.safe_dump(dict(path=str(DATA), train='images/train', val='images/val', names=NAMES)))
    print(json.dumps(manifest['counts'], indent=2))


def verify():
    manifest = load(DATA / 'manifest.json')
    validate_configuration(DATA, manifest['records'])
    assert sha(SOURCE / 'manifest.json') == manifest['source_manifest_sha']
    assert sha(ROOT / 'data/utensil-focus/coverage-reviewed-v7.json') == manifest['coverage_sha']
    assert sha(ROOT / 'data/utensil-focus/target-coverage-reviews-v7.json') == manifest['review_sha']
    if manifest.get('meal_scene_reviews_sha'):
        assert sha(ROOT/'data/utensil-packet-focus/meal-scene-reviews-v10.json') == manifest['meal_scene_reviews_sha']
    if manifest.get('extra_reviews_sha'):
        assert sha(ROOT/'data/utensil-packet-focus/extra-reviews-v3.json')==manifest['extra_reviews_sha']
    if manifest.get('negative_reviews_sha'):
        assert sha(ROOT/'data/utensil-packet-focus/negative-reviews-v4.json')==manifest['negative_reviews_sha']
        assert sha(ROOT/'data/utensil-packet-focus/packet-polygons-v4.json')==manifest['polygon_reviews_sha']
        assert sha(ROOT/'data/utensil-packet-focus/training-corrections-v4.json')==manifest['training_corrections_sha']
    assert hashlib.sha256(json.dumps(manifest['records'], sort_keys=True).encode()).hexdigest() == manifest['fingerprint']
    validate(manifest['records'], DATA)
    assert counts(manifest['records']) == manifest['counts']
    return manifest


def initialize():
    import torch
    from ultralytics import YOLO, __version__
    verify()
    if COCO_RESTART or HEAD_ONLY:
        dest = RUN/'initial.pt'
        if dest.exists():
            raise ValueError(f'Checkpoint exists: {dest}')
        source_weights = (ROOT/'model/utensil-packet-focus/real-only-v7/best.pt' if BALANCED_REAL or MEAL_SCENES else
                          ROOT/'model/utensil-packet-focus/head-only-v6/train/weights/last.pt' if REAL_ONLY else
                          ROOT/'model/utensil-packet-focus/coco-restart-v5/best.pt'
                          if HEAD_ONLY else ROOT/'model/weights/yolo11n.pt')
        if HEAD_ONLY:
            report_path = (source_weights.parent/'selected.json' if BALANCED_REAL or MEAL_SCENES else
                           ROOT/'model/utensil-packet-focus/head-only-v6/latest.json' if REAL_ONLY else source_weights.parent/'selected.json')
            assert sha(source_weights) == load(report_path)['weights_sha']
        torch.manual_seed(42)
        source_model = YOLO(str(source_weights)).model.float().cpu()
        model = copy.deepcopy(source_model) if HEAD_ONLY else coco_restart_model(source_model)
        assert model.names == dict(enumerate(NAMES))
        RUN.mkdir(parents=True, exist_ok=True)
        torch.save(dict(model=model.half(), ema=None, optimizer=None, epoch=-1,
                        version=__version__, train_args=dict(task='detect', data=str(DATA/'data.yaml'), epochs=80)), dest)
        save(RUN/'initialization.json', dict(source=str(source_weights), source_sha=sha(source_weights),
             output_sha=sha(dest), fresh_two_class_classifier=not HEAD_ONLY,
             freeze_backbone=HEAD_ONLY, dataset_fingerprint=verify()['fingerprint']))
        return
    source_weights = WEIGHTS
    if BALANCED:
        previous_name = 'expanded-v3' if CONTEXT else 'balanced-v2' if EXPANDED else 'warm-low-v1'
        previous_run = ROOT/'model/utensil-packet-focus'/previous_name
        source_weights = previous_run/'best.pt' if EXPANDED else previous_run/'train/weights/last.pt'
        assert sha(source_weights) == load(previous_run/('selected.json' if EXPANDED else 'latest.json'))['weights_sha']
        previous_data = load(ROOT/'data/utensil-packet-focus'/('expanded-v3' if CONTEXT else 'balanced-v2' if EXPANDED else 'reviewed-v1')/'manifest.json')
        assert [r for r in previous_data['records'] if r['split']=='val'] == [
            r for r in load(DATA/'manifest.json')['records'] if r['split']=='val']
    else:
        assert sha(source_weights) == WEIGHTS_SHA
    prior = load(ROOT/'data/utensil-focus/grouped-v6-packets/manifest.json')
    current = load(SOURCE/'manifest.json')
    def validation_identity(manifest):
        return [(r['name'], r['sha'], r['pixels'], r['group'], r.get('capture'),
                 [row for row in r['rows'] if int(row.split()[0]) in (3, 7)])
                for r in manifest['records'] if r['split']=='val']
    if validation_identity(prior) != validation_identity(current):
        raise ValueError('Source checkpoint validation lineage changed')
    dest = RUN / 'initial.pt'
    if dest.exists():
        raise ValueError(f'Checkpoint exists: {dest}')
    original = YOLO(str(source_weights)).model.float().cpu().eval()
    projected = (copy.deepcopy(original) if BALANCED else project_model(original)).eval()
    assert list(projected.names.values()) == NAMES
    with torch.no_grad():
        x = torch.rand(1, 3, 64, 64)
        old, new = original(x)[0], projected(x)[0]
        torch.testing.assert_close(old[:, :4], new[:, :4], rtol=0, atol=0)
        torch.testing.assert_close(old[:, 4:] if BALANCED else old[:, [7, 11]], new[:, 4:], rtol=1e-5, atol=1e-7)
    RUN.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=projected.half(), ema=None, optimizer=None, epoch=-1,
                    version=__version__, train_args=dict(task='detect', data=str(DATA/'data.yaml'), epochs=80)), dest)
    save(RUN / 'initialization.json', dict(source=str(source_weights), source_sha=sha(source_weights),
         output_sha=sha(dest), decoded_box_and_score_equivalence=True))


def evaluate(weights, tag, split='val', imgsz=640, augment=False):
    from ultralytics import YOLO
    manifest = verify()
    digest = sha(weights)
    model = YOLO(str(weights))
    assert list(model.names.values()) == NAMES
    metrics = model.val(data=str(DATA/'data.yaml'), split=split, imgsz=imgsz, batch=1 if augment or imgsz>640 else 4, device='mps', augment=augment,
                        workers=0, plots=False, project=str(RUN/'evaluations'), name=tag, exist_ok=False)
    report = dict(weights=str(weights), weights_sha=digest, split=split, dataset_fingerprint=manifest['fingerprint'],
                  independent_test=False, imgsz=imgsz, test_time_augmentation=augment, classes={}, overall=metrics.results_dict)
    for i, n in enumerate(NAMES):
        p, r, ap50, ap95 = metrics.box.class_result(i)
        report['classes'][n] = dict(precision=float(p), recall=float(r), AP50=float(ap50), AP50_95=float(ap95))
    assert sha(weights) == digest
    verify()
    save(RUN / f'{tag}.json', report)
    print(json.dumps(report, indent=2))
    return report


def packet_selection_score(scores):
    """Improve the weaker class without losing the 50% utensil milestone."""
    values = [float(x) for x in scores]
    if len(values) != 2 or not all(math.isfinite(x) for x in values) or values[0] < .5:
        return -1.
    return min(values)


def train(minutes, resume):
    """Fresh AdamW on learned head; native full checkpoint supports 80-total resume."""
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    verify()
    baseline_path = RUN / 'baseline.json'
    if not baseline_path.exists():
        evaluate(RUN/'initial.pt', 'baseline')
    baseline = load(baseline_path)
    selected = RUN / 'selection.json'
    if not selected.exists():
        shutil.copy2(RUN/'initial.pt', RUN/'best.pt')
        save(selected, dict(epoch=0, min_AP50=min(c['AP50'] for c in baseline['classes'].values())))
    class Trainer(DetectionTrainer):
        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
            return super().get_dataloader(dataset_path, 4 if mode=='val' else batch_size, rank, mode)
        def final_eval(self):
            # Fresh saved-model evaluation below replaces native repeated evaluation
            # and leaves the full native optimizer/EMA checkpoint resumable.
            pass
    started = time.monotonic()
    def completed(trainer):
        if trainer.epoch + 1 != len(trainer.csv.read_text().splitlines())-1:
            return  # Native final best-checkpoint evaluation is not a training epoch.
        shutil.copy2(trainer.last, RUN/'last_resume.pt')
        scores = trainer.validator.metrics.box.ap50
        score = packet_selection_score(scores) if BALANCED_REAL or MEAL_SCENES else min(float(v) for v in scores)
        if score > load(selected)['min_AP50']:
            checkpoint = torch.load(trainer.last, map_location='cpu', weights_only=False)
            torch.save(checkpoint, RUN/'best.pt')
            save(selected, dict(epoch=trainer.epoch+1, min_AP50=score))
        if minutes and time.monotonic()-started >= 60*minutes:
            trainer.stop = True
    if not resume and (RUN/'train').exists():
        raise ValueError('Training directory exists; use --resume')
    model = YOLO(str(RUN/'last_resume.pt' if resume else RUN/'initial.pt'))
    model.add_callback('on_model_save', completed)
    kwargs = dict(data=str(DATA/'data.yaml'), epochs=80, batch=4, imgsz=640, device='mps',
                  workers=0, optimizer='AdamW', lr0=.0003, warmup_bias_lr=0., warmup_epochs=3.,
                  nbs=64, seed=42, patience=0, plots=False, project=str(RUN), name='train',
                  exist_ok=False, save=True)
    if HEAD_ONLY:
        kwargs['freeze'] = 23  # Native trainer also holds frozen BatchNorm statistics fixed.
    model.train(trainer=Trainer, **({'resume':True} if resume else kwargs))
    del model
    import gc
    gc.collect()
    torch.mps.empty_cache()
    evaluate(RUN/'train/weights/last.pt', 'latest')
    evaluate(RUN/'best.pt', 'selected')


def main():
    global DATA, RUN, BALANCED, EXPANDED, CONTEXT, COCO_RESTART, HEAD_ONLY, REAL_ONLY, BALANCED_REAL, MEAL_SCENES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build','verify','initialize','train','evaluate'])
    parser.add_argument('--minutes', type=float, default=0,
                        help='Training budget checked after completed epochs; saved-model evaluation follows. 0 runs all 80 total epochs.')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--split', choices=['train','val'], default='val')
    parser.add_argument('--eval-imgsz', type=int, default=640, help='Evaluation resolution; training remains 640.')
    parser.add_argument('--tta', action='store_true', help='Use Ultralytics multi-scale/flip inference for evaluation.')
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument('--balanced-packets', action='store_true', help='Continuation: cap the repeated legacy packet session at four photos.')
    variants.add_argument('--expanded-packets', action='store_true', help='Capped session plus eight fully reviewed mayonnaise photos; warm-start from balanced-v2.')
    variants.add_argument('--packet-context', action='store_true', help='Reviewed hard negatives plus masked packet context/scale augmentation; frozen real validation.')
    variants.add_argument('--coco-restart', action='store_true', help='Fresh COCO initialization on the unchanged context-v4 dataset.')
    variants.add_argument('--head-only', action='store_true', help='Train only the detection head from the selected COCO-restart checkpoint.')
    variants.add_argument('--real-only', action='store_true', help='Head-only continuation using reviewed real photos; remove synthetic training composites.')
    variants.add_argument('--balanced-real', action='store_true', help='Head-only continuation with fourfold sampling of real packet-positive training photos.')
    variants.add_argument('--meal-scenes', action='store_true', help='Real-only continuation with three newly reviewed meal photos; no sample repetition.')
    args = parser.parse_args()
    if any((args.balanced_packets, args.expanded_packets, args.packet_context, args.coco_restart,
            args.head_only, args.real_only, args.balanced_real, args.meal_scenes)):
        BALANCED = True
        COCO_RESTART = args.coco_restart
        BALANCED_REAL = args.balanced_real
        MEAL_SCENES = args.meal_scenes
        REAL_ONLY = args.real_only or BALANCED_REAL or MEAL_SCENES
        HEAD_ONLY = args.head_only or REAL_ONLY
        CONTEXT = args.packet_context or COCO_RESTART or HEAD_ONLY
        EXPANDED = args.expanded_packets or CONTEXT
        DATA = ROOT/'data/utensil-packet-focus'/('context-v4' if CONTEXT else 'expanded-v3' if EXPANDED else 'balanced-v2')
        RUN = ROOT/'model/utensil-packet-focus'/('context-v4' if CONTEXT else 'expanded-v3' if EXPANDED else 'balanced-v2')
        if COCO_RESTART:
            RUN = ROOT/'model/utensil-packet-focus/coco-restart-v5'
        if HEAD_ONLY:
            RUN = ROOT/'model/utensil-packet-focus/head-only-v6'
        if REAL_ONLY:
            DATA = ROOT/'data/utensil-packet-focus/real-only-v7'
            RUN = ROOT/'model/utensil-packet-focus/real-only-v7'
        if BALANCED_REAL:
            DATA = ROOT/'data/utensil-packet-focus/balanced-real-v8'
            RUN = ROOT/'model/utensil-packet-focus/balanced-real-v8'
        if MEAL_SCENES:
            DATA = ROOT/'data/utensil-packet-focus/meal-scenes-v10'
            RUN = ROOT/'model/utensil-packet-focus/meal-scenes-v10'
    if args.weights is None:
        args.weights = RUN/'best.pt'
    if args.action == 'build': build()
    elif args.action == 'verify': print(json.dumps(verify()['counts'], indent=2))
    elif args.action == 'initialize': initialize()
    elif args.action == 'evaluate':
        tag = 'evaluation' if args.split=='val' else 'training-diagnostic'
        if args.tta or args.eval_imgsz != 640:
            tag += f'-{args.eval_imgsz}' + ('-tta' if args.tta else '')
        evaluate(args.weights, tag, args.split, args.eval_imgsz, args.tta)
    else: train(args.minutes, args.resume)


if __name__ == '__main__':
    main()
