"""Continue the best head-only model with the later feature layers unfrozen."""
from pathlib import Path
import importlib.util
import gc
import shutil
import torch
from ultralytics import YOLO

ROOT = Path.cwd()
RUN = Path(__file__).resolve().parent
PREVIOUS = RUN.parent / 'plastic-specialist-s-v2'
spec = importlib.util.spec_from_file_location('bottle_experiment', PREVIOUS / 'run.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.RUN = RUN
s = m.s

if __name__ == '__main__':
    initial = RUN / 'initial.pt'
    if not initial.exists():
        shutil.copy2(PREVIOUS / 'best_bottle.pt', initial)
        s.save(RUN / 'initialization.json', dict(source=str(PREVIOUS / 'best_bottle.pt'),
                   weights_sha=s.sha(initial), source_selection=s.load(PREVIOUS / 'selection.json')))
    if not (RUN / 'selection.json').exists():
        baseline = m.evaluate(initial, 'baseline')
        m.evaluate(initial, 'baseline-tta640', True)
        shutil.copy2(initial, RUN / 'best_bottle.pt')
        s.save(RUN / 'selection.json', dict(epoch=0, AP50=baseline['AP50'], weights_sha=s.sha(initial)))
    experiment = dict(s.load(PREVIOUS / 'experiment.json'))
    experiment.update(freeze=10, previous_run=str(PREVIOUS), hypothesis='Adapting the later feature layers improves small and crushed bottle detection beyond the frozen-backbone plateau', selection='Highest ordinary validation AP50, initialization retained as incumbent')
    s.save(RUN / 'experiment.json', experiment)

    def completed(trainer):
        score = float(trainer.validator.metrics.box.ap50[0])
        print('BOTTLE_EPOCH', trainer.epoch + 1, 'AP50', score, flush=True)
        shutil.copy2(trainer.last, RUN / 'last_resume.pt')
        if score > s.load(RUN / 'selection.json')['AP50']:
            shutil.copy2(trainer.last, RUN / 'best_bottle.pt')
            s.save(RUN / 'selection.json', dict(epoch=trainer.epoch + 1, AP50=score, weights_sha=s.sha(RUN / 'best_bottle.pt')))
        if score >= .85 or (RUN / 'stop-after-epoch').exists():
            trainer.stop = True

    resume = (RUN / 'last_resume.pt').exists()
    model = YOLO(str(RUN / 'last_resume.pt' if resume else initial))
    model.add_callback('on_model_save', completed)
    model.train(trainer=m.Trainer, resume=resume, data=str(m.DATA / 'data.yaml'), classes=[0],
                epochs=40, batch=4, imgsz=640, device='mps', workers=0,
                optimizer='AdamW', lr0=.0001, warmup_bias_lr=0., warmup_epochs=1.,
                nbs=16, seed=42, patience=0, freeze=10, plots=False, project=str(RUN),
                name='train', save=True, mosaic=.5, close_mosaic=10)
    del model
    gc.collect()
    torch.mps.empty_cache()
    m.evaluate(RUN / 'best_bottle.pt', 'selected')
    m.evaluate(RUN / 'best_bottle.pt', 'selected-tta640', True)
    s.validate(m.MANIFEST['records'], m.DATA)
