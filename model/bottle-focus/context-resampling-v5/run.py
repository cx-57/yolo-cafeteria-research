"""Oversample existing training context positives; preserve fixed validation."""
from pathlib import Path
import importlib.util
import gc
import os
import shutil
import yaml
import torch
from ultralytics import YOLO

RUN = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('bottle_common', RUN.parent / 'plastic-specialist-s-v2/run.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.RUN = RUN
s = m.s
SOURCE = RUN.parent / 'plastic-specialist-s-v3/initial.pt'
assert s.sha(SOURCE) == s.load(RUN.parent / 'plastic-specialist-s-v3/baseline.json')['weights_sha']

if __name__ == '__main__':
    train = [r for r in m.MANIFEST['records'] if r['split'] == 'train']
    exposures = []
    repeated = []
    for record in train:
        context = bool(record['rows']) and record['name'].startswith(('trash_detection__', 'plate__', 'external_bottle__'))
        count = 4 if context else 1
        exposures.extend([str(m.DATA / 'images/train' / record['name'])] * count)
        if context:
            repeated.append(record['name'])
    assert len(set(exposures)) == 1056 and len(exposures) == 1428 and len(repeated) == 124
    listing = RUN / 'training-exposures.txt'
    listing.write_text('\n'.join(exposures) + '\n')
    config = RUN / 'data.yaml'
    config.write_text(yaml.safe_dump(dict(path=str(m.DATA), train=str(listing), val=str(m.DATA / 'images/val'), names=['bottle']), sort_keys=False))
    experiment = dict(s.load(RUN.parent / 'plastic-specialist-s-v2/experiment.json'))
    experiment.update(source=str(SOURCE), source_sha=s.sha(SOURCE),
                      training_unique_images=1056, training_exposures_per_epoch=1428,
                      training_bottle_objects_unique=356, training_bottle_exposures_per_epoch=890,
                      context_positive_repeats=4, repeated_training_photos=repeated,
                      hypothesis='More context-positive exposure improves small-bottle ranking and recall; all negative photos remain included once per epoch',
                      validation_images_and_labels_unchanged=True)
    s.save(RUN / 'experiment.json', experiment)
    if not (RUN / 'selection.json').exists():
        shutil.copy2(SOURCE, RUN / 'best_bottle.pt')
        s.save(RUN / 'selection.json', dict(epoch=0, AP50=s.load(RUN.parent / 'plastic-specialist-s-v3/baseline.json')['AP50'], weights_sha=s.sha(SOURCE)))
    resume = (RUN / 'last_resume.pt').exists()
    phase_start = s.load(RUN / 'progress.json')['epoch'] if resume else 0
    phase_epochs = int(os.environ.get('BOTTLE_PHASE_EPOCHS', '8'))

    def completed(trainer):
        score = float(trainer.validator.metrics.box.ap50[0])
        print('BOTTLE_EPOCH', trainer.epoch + 1, 'AP50', score, flush=True)
        shutil.copy2(trainer.last, RUN / 'last_resume.pt')
        s.save(RUN / 'progress.json', dict(epoch=trainer.epoch + 1, AP50=score))
        if trainer.epoch >= 8:
            current = {key: value.detach().cpu().clone() for key, value in trainer.ema.ema.state_dict().items()}
            path = RUN / 'averaged-state.pt'
            if path.exists():
                average = torch.load(path, map_location='cpu', weights_only=True)
                average['count'] += 1
                for key, value in current.items():
                    if value.is_floating_point():
                        average['state'][key].lerp_(value.float(), 1 / average['count'])
                    else:
                        average['state'][key].copy_(value)
            else:
                average = dict(count=1, first_epoch=trainer.epoch + 1,
                               state={key: value.float() if value.is_floating_point() else value for key, value in current.items()})
            average['last_epoch'] = trainer.epoch + 1
            torch.save(average, RUN / 'averaged-state.tmp.pt')
            os.replace(RUN / 'averaged-state.tmp.pt', path)
            del current, average
        if score > s.load(RUN / 'selection.json')['AP50']:
            shutil.copy2(trainer.last, RUN / 'best_bottle.pt')
            s.save(RUN / 'selection.json', dict(epoch=trainer.epoch + 1, AP50=score, weights_sha=s.sha(RUN / 'best_bottle.pt')))
        if score >= .85 or trainer.epoch + 1 >= phase_start + phase_epochs or (RUN / 'stop-after-epoch').exists():
            trainer.stop = True

    model = YOLO(str(RUN / 'last_resume.pt' if resume else SOURCE))
    model.add_callback('on_model_save', completed)
    model.train(trainer=m.Trainer, resume=resume, data=str(config), classes=[0],
                epochs=40, batch=4, imgsz=640, device='mps', workers=0,
                optimizer='AdamW', lr0=.0001, warmup_bias_lr=0., warmup_epochs=1.,
                nbs=16, seed=42, patience=0, freeze=23, plots=False, project=str(RUN),
                name='train', save=True, mosaic=.5, close_mosaic=10)
    del model
    gc.collect()
    torch.mps.empty_cache()
    m.evaluate(RUN / 'best_bottle.pt', 'selected')
    m.evaluate(RUN / 'best_bottle.pt', 'selected-tta640', True)
    if (RUN / 'averaged-state.pt').exists():
        average = torch.load(RUN / 'averaged-state.pt', map_location='cpu', weights_only=True)
        averaged_model = YOLO(str(SOURCE)).model.float().cpu()
        averaged_model.load_state_dict(average['state'], strict=True)
        averaged_model.eval()
        weights = RUN / 'averaged-bottle.pt'
        torch.save(dict(model=averaged_model.half(), ema=None, optimizer=None, epoch=-1,
                        version=m.__version__, train_args=dict(task='detect')), weights)
        s.save(RUN / 'averaging.json', dict(count=average['count'], first_epoch=average['first_epoch'],
                   last_epoch=average['last_epoch'], weights_sha=s.sha(weights),
                   method='Arithmetic mean of saved epoch EMA parameters; BatchNorm running statistics remain fixed'))
        del average, averaged_model
        gc.collect()
        m.evaluate(weights, 'averaged')
        m.evaluate(weights, 'averaged-tta640', True)
    s.validate(m.MANIFEST['records'], m.DATA)
