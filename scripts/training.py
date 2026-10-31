"""Train a separate seven-class YOLO candidate with native, label-aware augmentation."""
import argparse
import copy
import json
import math
import shutil
import tempfile
from pathlib import Path
import yaml
from PIL import Image
from specialists import NAMES, sha
from storage import ROOT

PROFILE = ROOT / 'configs/train-angles.yaml'
SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
TRAIN_KEYS = {'epochs', 'imgsz', 'batch', 'nbs', 'workers', 'seed', 'cache', 'plots',
              'optimizer', 'lr0', 'lrf', 'warmup_epochs', 'warmup_bias_lr', 'patience', 'freeze'}
AUGMENT_KEYS = {'degrees', 'shear', 'perspective', 'translate', 'scale', 'fliplr', 'flipud',
                'hsv_h', 'hsv_s', 'hsv_v', 'mosaic', 'mixup', 'cutmix', 'copy_paste'}


def load_profile(path=PROFILE):
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict) or value.get('names') != NAMES:
        raise ValueError('Training profile class mapping must match the seven retained classes')
    if set(value) != {'names', 'training', 'augmentation'} or not isinstance(value['training'], dict) or not isinstance(value['augmentation'], dict):
        raise ValueError('Invalid training profile structure')
    if set(value['training']) - TRAIN_KEYS or set(value['augmentation']) - AUGMENT_KEYS:
        raise ValueError('Unsupported profile setting: output, resume and class scope are controlled by the command')
    if not {'epochs', 'imgsz', 'seed'} <= set(value['training']):
        raise ValueError('Training profile needs epochs, imgsz and seed')
    return value


def inspect_dataset(path, require_all_classes=True):
    path = Path(path).expanduser().resolve()
    data = yaml.safe_load(path.read_text())
    allow_cropped_food_train = data.get('allow_cropped_food_train', False)
    if not isinstance(allow_cropped_food_train, bool):
        raise ValueError('allow_cropped_food_train must be a boolean')
    names = data.get('names')
    if isinstance(names, dict):
        names = [names.get(i, names.get(str(i))) for i in range(len(names))]
    if names != NAMES or data.get('nc', len(NAMES)) != len(NAMES):
        raise ValueError('Dataset class mapping must match the seven retained classes')
    root = Path(data.get('path', path.parent)).expanduser()
    root = root if root.is_absolute() else path.parent / root
    resolved = dict(data, path=str(root.resolve()), names=NAMES.copy(), nc=len(NAMES))
    records, missing, seen = {}, {}, {}
    for split in ['train', 'val', 'test']:
        entries = data.get(split)
        if not entries:
            if split != 'test':
                raise ValueError('Dataset needs train and val splits')
            continue
        entries = entries if isinstance(entries, list) else [entries]
        locations = [Path(x).expanduser() if Path(x).expanduser().is_absolute() else root/x for x in entries]
        resolved[split] = [str(x.resolve()) for x in locations]
        images = []
        for location in locations:
            if location.is_dir():
                images.extend(p for p in location.rglob('*') if p.suffix.lower() in SUFFIXES)
            elif location.suffix == '.txt' and location.is_file():
                images.extend((Path(line) if Path(line).is_absolute() else location.parent/line).resolve()
                              for line in location.read_text().splitlines() if line.strip())
            else:
                raise ValueError('Expected image directory or image-list file: ' + str(location))
        if not images:
            raise ValueError('Empty dataset split: ' + split)
        records[split], classes = [], set()
        for image in sorted(images):
            try:
                with Image.open(image) as im:
                    image_format = im.format
                    im.verify()
                if image_format == 'JPEG':
                    with image.open('rb') as handle:
                        handle.seek(-2, 2)
                        if handle.read() != b'\xff\xd9':
                            raise ValueError('JPEG needs repair: ' + str(image))
            except (OSError, SyntaxError) as exc:
                raise ValueError('Corrupt image: ' + str(image)) from exc
            digest = sha(image)
            if digest in seen:
                raise ValueError('Image overlap/duplicate between dataset records: ' + str(image))
            seen[digest] = split
            parts = list(image.parts)
            if 'images' in parts:
                parts[len(parts)-1-parts[::-1].index('images')] = 'labels'
            label = Path(*parts).with_suffix('.txt')
            if not label.is_file():
                raise ValueError('Explicit label file required, including negatives: ' + str(label))
            for row in label.read_text().splitlines():
                if not row.strip():
                    continue
                values = list(map(float, row.split()))
                if len(values) != 5 or not all(math.isfinite(v) for v in values):
                    raise ValueError('Invalid detection label: ' + str(label))
                c, x, y, w, h = values
                if c != int(c) or not 0 <= c < 7 or w <= 0 or h <= 0 or min(x-w/2,y-h/2) < -1e-6 or max(x+w/2,y+h/2) > 1+1e-6:
                    raise ValueError('Invalid class/box: ' + str(label))
                if c == 0 and not (split == 'train' and allow_cropped_food_train) and (min(x-w/2,y-h/2) <= 1e-6 or max(x+w/2,y+h/2) >= 1-1e-6):
                    raise ValueError('Cut-off source food must be reviewed/excluded: ' + str(image))
                classes.add(int(c))
            records[split].append(dict(image=str(image), sha256=digest, label=str(label), label_sha256=sha(label)))
        missing[split] = [name for i,name in enumerate(NAMES) if i not in classes]
        if require_all_classes and split in ['train', 'val'] and missing[split]:
            raise ValueError(split + ' needs ground-truth support for all seven classes: ' + ', '.join(missing[split]))
    return dict(data=resolved, records=records, missing_classes=missing, source_yaml=str(path), source_yaml_sha256=sha(path))


def snapshot_dataset(info, output):
    """Freeze audited membership and bytes; native caches/repairs cannot touch sources."""
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    result = copy.deepcopy(info)
    result['native_data'] = dict(result['data'], path=str(root))
    result['snapshot_records'] = {}
    for split, records in info['records'].items():
        result['snapshot_records'][split] = []
        for i, record in enumerate(records):
            image = root/'images'/split/(f'{i:06d}-' + Path(record['image']).name)
            label = root/'labels'/split/image.with_suffix('.txt').name
            image.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(record['image'], image)
            shutil.copyfile(record['label'], label)
            if sha(image) != record['sha256'] or sha(label) != record['label_sha256']:
                raise ValueError('Source changed while creating training snapshot')
            result['snapshot_records'][split].append(dict(record, image=str(image), label=str(label)))
        result['native_data'][split] = str(root/'images'/split)
    return result


def native_dataset(info, split, profile):
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    settings = dict(copy.deepcopy(profile['training']), **copy.deepcopy(profile['augmentation']))
    temporary = None
    if 'native_data' not in info:
        temporary = tempfile.TemporaryDirectory()
        info = snapshot_dataset(info, Path(temporary.name)/'input')
    dataset = YOLODataset(img_path=info['native_data'][split], data=info['native_data'], imgsz=settings['imgsz'],
                          batch_size=1, augment=split == 'train', hyp=get_cfg(overrides=settings),
                          rect=False, cache=False, prefix=split + ': ')
    dataset._temporary_snapshot = temporary
    return dataset


def verify_inputs(info):
    if sha(info['source_yaml']) != info['source_yaml_sha256']:
        raise ValueError('Dataset YAML changed during operation')
    for records in list(info['records'].values()) + list(info.get('snapshot_records', {}).values()):
        for r in records:
            if sha(r['image']) != r['sha256'] or sha(r['label']) != r['label_sha256']:
                raise ValueError('Input photo/label changed during operation')


def preview(info, profile, output, count):
    import random
    import numpy as np
    from PIL import Image, ImageDraw
    random.seed(profile['training']['seed']); np.random.seed(profile['training']['seed'])
    dataset = native_dataset(info, 'train', profile)
    indices, covered = [], set()
    for index, label in enumerate(dataset.labels):
        classes = set(int(c) for c in label['cls'].reshape(-1))
        if classes - covered:
            indices.append(index)
            covered.update(classes)
    indices.extend(i for i in range(len(dataset)) if i not in indices)
    examples = []
    for i in range(count):
        item = dataset[indices[i % len(indices)]]
        image = Image.fromarray(item['img'].permute(1, 2, 0).numpy())
        draw = ImageDraw.Draw(image); width,height = image.size
        boxes = item['bboxes'].tolist(); classes = item['cls'].reshape(-1).tolist()
        for c,(x,y,w,h) in zip(classes, boxes):
            if not (0 <= x <= 1 and 0 <= y <= 1 and w > 0 and h > 0):
                raise ValueError('Native augmentation produced invalid boxes')
            rect=((x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height)
            draw.rectangle(rect,outline='lime',width=2);draw.text(rect[:2],NAMES[int(c)],fill='lime')
        target=output/f'example-{i:02d}.jpg';image.save(target)
        examples.append(dict(source=item['im_file'],preview=str(target),classes=classes,boxes=boxes))
    (output/'examples.json').write_text(json.dumps(examples,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['train','preview'])
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile',type=Path,default=PROFILE)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--device',choices=['cpu','mps'],default='mps')
    parser.add_argument('--smoke-test',action='store_true',help='Force one functional epoch; allow missing-class smoke fixtures')
    parser.add_argument('--count',type=int,default=12)
    args=parser.parse_args()
    args.output=args.output.expanduser().resolve()
    if args.model is not None:
        args.model=args.model.expanduser().resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.epochs is not None and args.epochs < 1 or args.count < 1:
        parser.error('Epochs/count must be positive')
    if args.mode == 'train' and (args.model is None or not args.model.is_file()):
        parser.error('Training requires an existing local --model checkpoint; no automatic downloads')
    profile=load_profile(args.profile)
    info=inspect_dataset(args.data,require_all_classes=args.mode=='train' and not args.smoke_test)
    args.output.mkdir(parents=True)
    info=snapshot_dataset(info,args.output/'input')
    recipe=dict(profile=profile,profile_sha256=sha(args.profile),code_sha256=sha(Path(__file__)),
                dataset=info,smoke_test=args.smoke_test,
                note='Separate candidate; no automatic promotion. Online augmentation is train-only; native YOLO transforms boxes.')
    try:
        if args.mode=='preview':
            preview(info,profile,args.output,args.count)
        else:
            import torch
            import ultralytics
            from ultralytics import YOLO
            torch.set_num_threads(2)
            resolved=args.output/'resolved-data.yaml';resolved.write_text(yaml.safe_dump(info['native_data']))
            settings=dict(profile['training'],**profile['augmentation'])
            settings.update(data=str(resolved),device=args.device,project=str(args.output.parent),
                            name=args.output.name,exist_ok=True,augment=False,
                            resume=False,single_cls=False,classes=None,fraction=1.0)
            settings['epochs']=1 if args.smoke_test else args.epochs or settings['epochs']
            recipe.update(model=str(args.model.resolve()),model_sha256=sha(args.model),applied_settings=settings)
            recipe['runtime']=dict(torch=torch.__version__,ultralytics=ultralytics.__version__)
            (args.output/'recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
            model=YOLO(str(args.model))
            if model.task != 'detect':
                raise ValueError('Training requires an object-detection checkpoint')
            for key in ['save_dir', 'resume']:
                model.overrides.pop(key, None)
            def progress(trainer):
                (args.output/'status.json').write_text(json.dumps(dict(state='training',epoch=trainer.epoch+1,cap=settings['epochs'])))
            model.add_callback('on_fit_epoch_end',progress)
            model.train(**settings)
            if sha(args.model)!=recipe['model_sha256']:
                raise ValueError('Initialization checkpoint was modified')
        verify_inputs(info)
        (args.output/'recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
        (args.output/'status.json').write_text(json.dumps(dict(state='complete',mode=args.mode,smoke_test=args.smoke_test,
                                                                input_files_unchanged=True,active_models_changed=False)))
    except Exception as exc:
        (args.output/'status.json').write_text(json.dumps(dict(state='failed',error=str(exc))))
        raise


if __name__=='__main__':
    main()
