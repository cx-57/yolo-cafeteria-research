"""Train the larger COCO-initialized YOLO11m bottle head on reviewed context-balanced data."""
from pathlib import Path
import gc
import importlib.util
import os
import shutil
import torch
from ultralytics import YOLO

RUN = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('bottle_common', RUN.parent / 'plastic-specialist-s-v2/run.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.RUN = RUN
s = m.s
DATA_CONFIG = RUN / 'data.yaml'
build_spec = importlib.util.spec_from_file_location('external_dataset', RUN / 'build.py')
dataset = importlib.util.module_from_spec(build_spec)
build_spec.loader.exec_module(dataset)
INITIAL = RUN / 'initial.pt'

if __name__ == '__main__':
    audit = dataset.verify()
    proof=s.load(RUN/'initialization.json')
    assert proof['architecture']=='YOLO11m' and s.sha(INITIAL)==proof['weights_sha']
    assert s.sha(Path(proof['source']))==proof['source_sha']
    if not (RUN / 'selection.json').exists():
        baseline = m.evaluate(INITIAL, 'baseline', batch=2)
        shutil.copy2(INITIAL, RUN / 'best_bottle.pt')
        s.save(RUN / 'selection.json', dict(epoch=0, AP50=baseline['AP50'], weights_sha=s.sha(INITIAL)))
    s.save(RUN / 'experiment.json', dict(target='bottle', hypothesis='A larger pretrained representation improves the attainable plastic-bottle specialist score',
               source=s.load(RUN / 'initialization.json'), data=str(DATA_CONFIG),
               source_dataset_fingerprint=m.MANIFEST['source_fingerprint'],
               validation_protocol=m.MANIFEST['validation_protocol'], validation_images=246,
               validation_bottle_objects=67, training_unique_images=audit['training_unique_images'], added_images=57, added_bottle_boxes=109, quality_holds=15, hard_negative_photos=12, hard_negative_repeats=1, external_review='AI visual review; not human verified',
               training_exposures_per_epoch=audit['training_exposures'], context_positive_repeats=8, external_positive_repeats=1,
               epochs=40, batch=2, imgsz=640, freeze=23, optimizer='AdamW',
               lr0=.0001, nbs=16, mosaic=.5, close_mosaic=0, scale=.5,
               normalization='All pretrained BatchNorm running statistics fixed', independent_test=False))
    resume = (RUN / 'last_resume.pt').exists()
    phase_start = s.load(RUN / 'progress.json')['epoch'] if resume else 0
    phase_epochs = int(os.environ.get('BOTTLE_PHASE_EPOCHS', '6'))

    def completed(trainer):
        score = float(trainer.validator.metrics.box.ap50[0])
        print('BOTTLE_EPOCH', trainer.epoch + 1, 'AP50', score, flush=True)
        shutil.copy2(trainer.last, RUN / 'last_resume.pt')
        s.save(RUN / 'progress.json', dict(epoch=trainer.epoch + 1, AP50=score))
        trained_selection = RUN / 'trained-selection.json'
        if not trained_selection.exists() or score > s.load(trained_selection)['AP50']:
            shutil.copy2(trainer.last, RUN / 'best_trained_bottle.pt')
            s.save(trained_selection, dict(epoch=trainer.epoch + 1, AP50=score, weights_sha=s.sha(RUN / 'best_trained_bottle.pt')))
        if score > s.load(RUN / 'selection.json')['AP50']:
            shutil.copy2(trainer.last, RUN / 'best_bottle.pt')
            s.save(RUN / 'selection.json', dict(epoch=trainer.epoch + 1, AP50=score, weights_sha=s.sha(RUN / 'best_bottle.pt')))
        if score >= .85 or trainer.epoch + 1 >= phase_start + phase_epochs or (RUN / 'stop-after-epoch').exists():
            trainer.stop = True

    model = YOLO(str(RUN / 'last_resume.pt' if resume else INITIAL))
    model.add_callback('on_model_save', completed)
    model.train(trainer=m.Trainer, resume=resume, data=str(DATA_CONFIG), classes=[0],
                epochs=40, batch=2, imgsz=640, device='mps', workers=0,
                optimizer='AdamW', lr0=.0001, warmup_bias_lr=0., warmup_epochs=1.,
                nbs=16, seed=42, patience=0, freeze=23, plots=False, project=str(RUN),
                name='train', save=True, mosaic=.5, close_mosaic=0, scale=.5)
    del model
    gc.collect()
    torch.mps.empty_cache()
    m.evaluate(RUN / 'best_bottle.pt', 'selected', batch=2)
    m.evaluate(RUN / 'best_bottle.pt', 'selected-tta640', True)
    dataset.verify()
