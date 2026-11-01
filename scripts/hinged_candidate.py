"""One controlled hinged-container specialist fine-tuning experiment.

Creates a separate 3-class candidate from existing SEVEN-class train labels.
Never trains on validation, test, public 44-image stress photos, or untouched
checkpoint/research freeze files. Development validation has just two hinged
instances: this experiment cannot prove independent accuracy improvements.
"""
import argparse
import copy
import hashlib
import json
import math
import random
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import yaml

import explore_candidates as frozen
import guard_candidate
import specialists as s
import training
from storage import ROOT, STORE, EVALUATION

OWNED = ('food', 'hinged_container', 'drink_carton')
MAP = {0: 0, 3: 1, 4: 2}
DEFAULT_DATA = STORE / 'data/combined-seven-school-oct29-20261031/data.yaml'
DEFAULT_BASELINE = ROOT / 'results/baseline.json'


def parse_rows(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        items = line.split()
        if len(items) != 5:
            raise ValueError('Non-5-column YOLO label: ' + str(path))
        c = int(items[0])
        if items[0] not in (str(c), str(float(c))):
            raise ValueError('Invalid class index: ' + line)
        values = [float(v) for v in items[1:]]
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Nonfinite YOLO box: ' + str(path))
        rows.append((c, values))
    return rows


def choose_train(records, seed=42, support_cap=250, background_cap=90):
    """Every hinged-positive training view, bounded food/carton positives,
    and plate-only negatives. No validation/test membership considered here."""
    rng = random.Random(seed)
    annotated = [(r, {c for c, _ in parse_rows(r['label'])}) for r in records]
    hinged = [r for r, classes in annotated if 3 in classes]
    if len(hinged) < 5:
        raise ValueError('Need >= 5 existing hinged-positive training views')
    selected = {r['image']:r for r in hinged}
    for cls in [4, 0]:
        options = [r for r, classes in annotated if cls in classes and r['image'] not in selected]
        rng.shuffle(options)
        selected.update({r['image']:r for r in options[:support_cap]})
    # Hard negatives: visible plates but none of the three owned classes.
    negatives = [r for r, classes in annotated if 1 in classes and not classes.intersection(MAP)]
    rng.shuffle(negatives)
    selected.update({r['image']:r for r in negatives[:background_cap]})
    chosen = sorted(selected.values(),key=lambda x:x['image'])
    return chosen, len(hinged), min(background_cap,len(negatives))


def export_dataset(info, out, repeat=3, support_cap=250, background_cap=90, seed=42):
    """Copy old training images/labels, remap owned classes, oversample ONLY
    hinged-positive training views (augmentation online; copies not new photos)."""
    if out.exists():
        raise FileExistsError('Refusing to reuse training snapshot')
    if not (1 <= repeat <= 5) or support_cap < 1 or background_cap < 0:
        raise ValueError('Invalid training repeat or caps')
    chosen, n_hinge, n_negative = choose_train(
        info['records']['train'],seed,support_cap,background_cap)
    val = info['records']['val']
    root = out
    counts = Counter()
    index = []
    for split, records in [('train', chosen), ('val', val)]:
        for i, r in enumerate(records):
            labels = parse_rows(r['label'])
            owned = [(MAP[c],values) for c, values in labels if c in MAP]
            repeats = repeat if split == 'train' and any(c == 1 for c,_ in owned) else 1
            for k in range(repeats):
                stem = f'{i:05d}_{k:02d}_{hashlib.sha256(r["image"].encode()).hexdigest()[:8]}'
                image = root / 'images' / split / (stem + Path(r['image']).suffix.lower())
                label = root / 'labels' / split / (stem + '.txt')
                image.parent.mkdir(parents=True,exist_ok=True)
                label.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(r['image'],image)
                if s.sha(image) != r['sha256']:
                    raise ValueError('Training source image changed during snapshot')
                label.write_text(''.join(
                    f'{c} ' + ' '.join(f'{v:.9g}' for v in values) + '\n'
                    for c,values in owned))
                for c,_ in owned:counts[(split, OWNED[c])] += 1
                index.append(dict(split=split, image=str(image), image_sha256=r['sha256'],
                                  original_image=r['image'],original_label_sha256=r['label_sha256'],
                                  repeat=k,classes=[OWNED[c] for c,_ in owned]))
    config = dict(path=str(root),train='images/train',val='images/val',nc=3,names=list(OWNED))
    (root/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    (root/'sampling.json').write_text(json.dumps(dict(
        original_train_selected=len(chosen),original_train_hinged_views=n_hinge,
        negative_plate_only_train_views=n_negative,hinge_repeat=repeat,
        selected_photos_are_not_independent= True,
        val_originals=len(val), # kept out of training
        annotation_boxes_by_split_and_class={f'{split}:{cls}':n for (split,cls),n in sorted(counts.items())},
        image_records=index),indent=2)+'\n')
    return dict(train_originals=len(chosen), train_hinged_views=n_hinge,
                train_hinged_oversampling=repeat, val_images=len(val), image_records=len(index))


def train(args, out):
    if out.exists():
        raise FileExistsError('Refusing to overwrite existing candidate experiment: '+str(out))
    protected=frozen.check_frozen(ROOT)
    data_path=Path(args.data).expanduser().resolve()
    info=training.inspect_dataset(data_path)
    if 'train' not in info['records'] or 'val' not in info['records']:
        raise ValueError('Expected training and validation splits')
    # No image can be in both roles. inspect_dataset also checks SHA duplicates.
    train_sha={r['sha256'] for r in info['records']['train']}
    val_sha={r['sha256'] for r in info['records']['val']}
    test_sha={r['sha256'] for r in info['records'].get('test',[])}
    if train_sha & (val_sha|test_sha):
        raise ValueError('Train shares exact image bytes with frozen evaluation split')
    if not args.bundle:
        raise ValueError('Pass --bundle of the 44 public diagnostic images to enforce the no-training-on-stress-photos SHA gate')
    bundle=Path(args.bundle).expanduser().resolve()
    with zipfile.ZipFile(bundle) as z:
        stress=json.loads(z.read('manifest.json'))
        excluded={r['sha256'] for r in stress['images']}
        if len(stress['images']) != 44 or len(excluded) != 44:
            raise ValueError('Expected the immutable 44-image source-screening bundle')
        for row in stress['images']:
            if hashlib.sha256(z.read(row['path'])).hexdigest()!=row['sha256']:
                raise ValueError('Stress image hash mismatch: '+row['path'])
    if train_sha & excluded:
        raise ValueError('Existing training split overlaps the 44-photo diagnostic cohort; do not train on these photos')
    out.mkdir(parents=True)
    summary=export_dataset(info,out/'data',repeat=args.repeat, seed=args.seed,
                           support_cap=args.support_cap, background_cap=args.background_cap)
    summary.update(dict(source_yaml=str(data_path),source_yaml_sha256=s.sha(data_path),stress_image_count=44,stress_train_overlap=0,
                        original_frozen_checkpoint_sha256=protected['weights/food-container-carton.pt'],
                        epochs=args.epochs,learning_rate=args.lr,
                        status='TRAINING_CANDIDATE_NOT_PROMOTED',
                        warning='Validation has only two hinged-container objects. No independent improvement claim.'))
    (out/'experiment.json').write_text(json.dumps(summary,indent=2)+'\n')
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(2)
    path=ROOT/'weights/food-container-carton.pt'
    model=YOLO(str(path))
    names=model.names
    if isinstance(names,dict):names=[names[i] for i in range(len(names))]
    if list(names) != list(OWNED):
        raise ValueError('Original checkpoint must have exactly three classes: '+str(OWNED))
    for field in ('resume','save_dir'):
        model.overrides.pop(field,None)
    model.train(data=str(out/'data/data.yaml'),device=args.device,
                project=str(out),name='fit',exist_ok=False,
                epochs=args.epochs,imgsz=640,batch=1,nbs=8,
                workers=0,seed=args.seed,optimizer='AdamW',lr0=args.lr,lrf=0.25,
                warmup_epochs=1,patience=args.epochs,freeze=10,
                degrees=10,translate=0.06,scale=0.20,shear=1.0,
                fliplr=0.5,flipud=0.0,hsv_h=.01,hsv_s=.20,hsv_v=.18,
                mosaic=0.0,mixup=0.0,copy_paste=0.0,plots=False,cache=False,
                val=True,save=True,verbose=True)
    best=out/'fit/weights/best.pt'
    if not best.is_file():raise FileNotFoundError('Candidate best.pt absent')
    if frozen.check_frozen(ROOT)!=protected:
        raise ValueError('Frozen checkpoints/config/policy changed after candidate run')
    config=s.read(s.CONFIG)
    config=copy.deepcopy(config)
    for spec in config['models']:
        if spec['id']=='food-container-carton':
            spec['weights']=str(best.resolve())
            spec['sha256']=s.sha(best)
    s.save(out/'candidate-specialists.json',config)
    summary.update(dict(status='TRAINED_NOT_PROMOTED',candidate_weight=str(best),
                        candidate_sha256=s.sha(best), active_models_unchanged=True))
    (out/'experiment.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def evaluate(args,out):
    config=out/'candidate-specialists.json'
    if not config.is_file():
        raise FileNotFoundError('Train the candidate first; missing candidate config: '+str(config))
    baseline=s.read(DEFAULT_BASELINE)
    manifest=Path(args.manifest).expanduser().resolve()
    if s.sha(manifest)!=baseline.get('manifest_sha256'):
        raise ValueError('Must compare to the SAME pinned 186-photo baseline validation manifest')
    protected=frozen.check_frozen(ROOT)
    source=s.read(manifest)
    records,classes=source['records'],source['evaluated_classes']
    s.validate_annotations(records,classes)
    s.verify_images(records)
    candidate_config=s.read(config)
    receipt=s.predict(records,args.device,config_path=config)
    receipt=s.validate_receipt(receipt,records,candidate_config,s.sha(config))
    scored=s.score_predictions(records,receipt['predictions'],classes)
    result=dict(scored, independent_test=False,manifest_sha256=s.sha(manifest),
                annotation_status='AI-assisted development validation, hinged class has only TWO labels',
                model_config_sha256=s.sha(config))
    gate=guard_candidate.check_candidate(
        baseline,result,target='hinged_container',minimum_gain=0.0,max_regression=0.005)
    if frozen.check_frozen(ROOT)!=protected:
        raise ValueError('Frozen checkpoints altered while evaluating')
    s.save(out/'candidate-validation.json',result)
    s.save(out/'validation-gate.json',gate)
    return dict(target='hinged_container',hinge_baseline_AP50_95=baseline['per_class']['hinged_container']['AP50_95'],
                hinge_candidate_AP50_95=result['per_class']['hinged_container']['AP50_95'],
                hinge_baseline_objects=2,gate_pass=gate['passed'],gate_blockers=gate['blockers'],
                recommendation='NOT AUTOMATICALLY PROMOTED. Need visual diagnostics and independent field evaluation.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['train','evaluate','stress'])
    p.add_argument('--data',type=Path,default=DEFAULT_DATA)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=EVALUATION)
    p.add_argument('--bundle',type=Path,help='Existing 44-image ZIP; required to exclude from training and for stress comparison')
    p.add_argument('--device',choices=['cpu','mps'],default='mps')
    p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--repeat',type=int,default=3)
    p.add_argument('--support-cap',type=int,default=250)
    p.add_argument('--background-cap',type=int,default=90)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--lr',type=float,default=0.00005)
    a=p.parse_args()
    if a.epochs<1 or not 1<=a.repeat<=5 or a.support_cap<1 or a.background_cap<0 or not 0<a.lr<0.1:
        p.error('Invalid experiment training settings')
    out=a.out.expanduser().resolve()
    if a.mode == 'train':
        result=train(a,out)
    elif a.mode == 'evaluate':
        result=evaluate(a,out)
    else:
        if not a.bundle:
            p.error('--bundle is required for stress mode')
        from compare_hinged_stress import run as compare_stress
        result=compare_stress(a.bundle.expanduser().resolve(),
                              out/'fit/weights/best.pt',
                              out/'stress-comparison',a.device)
        result={k:v for k,v in result.items() if k!='per_image'}
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
