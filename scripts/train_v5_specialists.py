"""Train explicitly requested V5 specialists and compare the complete six-class hybrid."""
import argparse
import gc
import json
from pathlib import Path

from v5_dataset import NAMES, sha, verify_dataset, write_json
from v5_projection import without_bottle, project_classes
from v5_specialists import (build_view, verify_view, evaluate_checkpoint, load_evaluation,
    accept_specialist, merge_owned_predictions, validation_records)
from v5_trials import ensure_disk_space
from specialists import score_predictions

TARGETS = ['utensil', 'drink_carton', 'wrapper']


def run(parent, dataset, output, epochs=5, device='0', batch=4, prepare_only=False):
    import torch
    from ultralytics import YOLO
    parent, dataset, output = Path(parent).resolve(), Path(dataset).resolve(), Path(output).resolve()
    data = verify_dataset(dataset)
    data_sha = sha(dataset / 'dataset_receipt.json')
    config = dict(source_checkpoint=str(parent), source_sha256=sha(parent), dataset_sha256=data_sha,
                  epochs=epochs, device=device, batch=batch, targets=TARGETS,
                  annotation_status=data['annotation_status'], test_evaluated=False,
                  authorization='User explicitly requested specialist trials')
    if epochs < 1 or batch < 1:
        raise ValueError('Epochs and batch must be positive')
    output.mkdir(parents=True, exist_ok=True)
    setup = output / 'setup.json'
    if setup.exists():
        if json.loads(setup.read_text()) != config:
            raise ValueError('Existing specialist experiment has different inputs or settings')
    else:
        write_json(setup, config)
    views = dataset.parent / (dataset.name + '-specialist-views-' + data_sha[:12])
    views.mkdir(exist_ok=True)
    base = output / 'unified-parent.pt'
    if not base.exists():
        model = YOLO(str(parent))
        names = list(model.names.values())
        if names == NAMES + ['bottle']:
            model.model = without_bottle(model.model)
        elif names != NAMES:
            raise ValueError('Parent must be the trained six/seven-class V5')
        model.ckpt = dict(epoch=-1, optimizer=None, ema=None,
                          train_args={'task':'detect','imgsz':800}, parent_sha256=sha(parent))
        model.save(str(base))
        write_json(output/'parent-projection.json',dict(source_sha256=sha(parent),sha256=sha(base)))
        del model; gc.collect()
    else:
        projection=json.loads((output/'parent-projection.json').read_text())
        if projection != dict(source_sha256=sha(parent),sha256=sha(base)):
            raise ValueError('Projected parent weights changed')
    for name in TARGETS:
        folder = output / name; folder.mkdir(exist_ok=True)
        view = views / name
        if not view.exists(): build_view(dataset, view, [NAMES.index(name)])
        else: verify_view(view)
        initial = folder / 'initial.pt'
        if not initial.exists():
            model = YOLO(str(base)); model.model = project_classes(model.model, [name])
            model.ckpt = dict(epoch=-1, optimizer=None, ema=None,
                train_args={'task':'detect','imgsz':800}, parent_sha256=sha(base))
            model.save(str(initial)); del model; gc.collect()
            write_json(folder/'initialization.json',dict(parent_sha256=sha(base),sha256=sha(initial),classes=[name]))
        else:
            initialization=json.loads((folder/'initialization.json').read_text())
            if initialization != dict(parent_sha256=sha(base),sha256=sha(initial),classes=[name]):
                raise ValueError('Specialist initial weights changed')
    if prepare_only:
        return dict(status='prepared_not_trained', output=str(output), models=TARGETS)
    ensure_disk_space(output, minimum_bytes=2 * 1024**3)
    base_view = views / 'unified'
    if not base_view.exists(): build_view(dataset, base_view, list(range(6)), include_train=False)
    baseline_dir = output / 'baseline-evaluation'
    if (baseline_dir / 'report.json').exists():
        baseline, base_predictions = load_evaluation(baseline_dir, base, base_view)
    else:
        baseline, base_predictions = evaluate_checkpoint(base, base_view, baseline_dir, device)
    selected = []
    decisions = {}
    models = []
    for name in TARGETS:
        folder = output / name; view = views / name
        best = folder / 'train/weights/best.pt'
        evaluation = folder / 'evaluation'
        if (evaluation / 'report.json').exists():
            scores, predictions = load_evaluation(evaluation, best, view)
        else:
            if (folder / 'train').exists():
                raise ValueError(f'{name} training is incomplete. Preserve its checkpoint for recovery; do not restart over it.')
            print(f'Training {name}: {epochs} epochs', flush=True)
            model = YOLO(str(folder / 'initial.pt'))
            expected = {k:v.detach().cpu().clone() for k,v in model.model.named_parameters()}
            def verify_transfer(trainer):
                actual = dict(trainer.model.named_parameters())
                if actual.keys() != expected.keys(): raise ValueError('Specialist model was rebuilt differently')
                for k,v in expected.items(): torch.testing.assert_close(actual[k].detach().cpu(),v,rtol=0,atol=0)
                expected.clear()
            def stable_bn(trainer):
                for layer in trainer.model.modules():
                    if isinstance(layer,torch.nn.BatchNorm2d): layer.eval()
            model.add_callback('on_train_start', verify_transfer)
            model.add_callback('on_train_batch_start', stable_bn)
            model.add_callback('on_train_batch_end', lambda trainer: ensure_disk_space(trainer.save_dir,minimum_bytes=1024**3))
            for cache in (view / 'labels').glob('*.cache'): cache.unlink()
            model.train(data=str(view/'data.yaml'), epochs=epochs, patience=epochs,
                device=device,batch=batch,imgsz=800,workers=0,optimizer='AdamW',lr0=.0001,
                lrf=.1,cos_lr=True,weight_decay=.0005,warmup_epochs=.5,warmup_bias_lr=0.,
                freeze=10,nbs=64,seed=57,deterministic=True,scale=.1,degrees=3,translate=.05,
                mosaic=.1,mixup=0,close_mosaic=min(2,epochs-1),fliplr=.5,flipud=0,
                hsv_s=.3,hsv_v=.2,cache=False,plots=False,save=True,save_period=-1,
                project=str(folder),name='train',exist_ok=False)
            if model.trainer.epoch+1 != epochs: raise ValueError('Specialist did not complete its epoch budget')
            del model; gc.collect()
            if device == 'mps': torch.mps.empty_cache()
            verify_view(view)
            scores, predictions = evaluate_checkpoint(best, view, evaluation, device)
        accepted = accept_specialist(baseline['per_class'],scores['per_class'],[name])
        decisions[name] = dict(accepted=accepted, scores=scores)
        if accepted:
            selected.append(dict(classes=[name], predictions=predictions))
            models.append(dict(weights=str(best),sha256=sha(best),classes=[name],source_classes=[name]))
        hybrid_predictions = merge_owned_predictions(base_predictions, selected)
        hybrid = score_predictions(validation_records(dataset),hybrid_predictions,set(range(6)))
        base_classes = [n for n in NAMES if n not in {e['classes'][0] for e in selected}]
        report = dict(baseline=baseline,hybrid=hybrid,decisions=decisions,
            models=[dict(weights=str(base),sha256=sha(base),classes=base_classes,source_classes=NAMES)]+models,
            source_sha256=sha(parent),dataset_sha256=data_sha,test_evaluated=False,
            scope='development comparison; original checkpoint training exposure unresolved',
            status='complete' if len(decisions)==3 else 'in_progress')
        temp=output/'comparison.tmp';temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(output/'comparison.json')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--device',default='0')
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args()
    print(json.dumps(run(a.parent,a.dataset,a.output,a.epochs,a.device,a.batch,a.prepare_only),indent=2))


if __name__=='__main__': main()
