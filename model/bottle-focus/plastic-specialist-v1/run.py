"""Recover the interrupted YOLO11x bottle experiment, preserving every epoch."""
from pathlib import Path
import importlib.util
import gc
import os
import shutil
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

RUN = Path(__file__).resolve().parent
COMMON = RUN.parent / 'plastic-specialist-s-v2/run.py'
spec = importlib.util.spec_from_file_location('bottle_common', COMMON)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.RUN = RUN
s = m.s
INITIAL = RUN.parent / 'validation-now/bottle-yolo11x.pt'
assert s.sha(INITIAL) == s.load(RUN / 'experiment.json')['source_sha']

class Trainer(m.Trainer):
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        return super().get_dataloader(dataset_path, 2 if mode == 'val' else batch_size, rank, mode)

if __name__ == '__main__':
    if not (RUN / 'selection.json').exists():
        baseline = m.evaluate(INITIAL, 'baseline', batch=2)
        shutil.copy2(INITIAL, RUN / 'best_bottle.pt')
        s.save(RUN / 'selection.json', dict(epoch=0, AP50=baseline['AP50']))
    resume = (RUN / 'last_resume.pt').exists()
    phase_start = s.load(RUN / 'progress.json')['epoch'] if resume else 0
    phase_epochs = int(os.environ.get('BOTTLE_PHASE_EPOCHS', '2'))

    def completed(trainer):
        score = float(trainer.validator.metrics.box.ap50[0])
        print('BOTTLE_EPOCH', trainer.epoch + 1, 'AP50', score, flush=True)
        shutil.copy2(trainer.last, RUN / 'last_resume.pt')
        s.save(RUN / 'progress.json', dict(epoch=trainer.epoch + 1, AP50=score))
        if score > s.load(RUN / 'selection.json')['AP50']:
            shutil.copy2(trainer.last, RUN / 'best_bottle.pt')
            s.save(RUN / 'selection.json', dict(epoch=trainer.epoch + 1, AP50=score,
                                               weights_sha=s.sha(RUN / 'best_bottle.pt')))
        if score >= .85 or trainer.epoch + 1 >= phase_start + phase_epochs or (RUN / 'stop-after-epoch').exists():
            trainer.stop = True

    model = YOLO(str(RUN / 'last_resume.pt' if resume else INITIAL))
    model.add_callback('on_model_save', completed)
    model.train(trainer=Trainer, resume=resume, data=str(m.DATA / 'data.yaml'), classes=[0],
                epochs=40, batch=2, imgsz=640, device='mps', workers=0,
                optimizer='AdamW', lr0=.0001, warmup_bias_lr=0., warmup_epochs=1.,
                nbs=16, seed=42, patience=0, freeze=23, plots=False, project=str(RUN),
                name='train-recovered', save=True, mosaic=.5, close_mosaic=10)
    del model
    gc.collect()
    torch.mps.empty_cache()
    m.evaluate(RUN / 'best_bottle.pt', 'selected', batch=2)
    m.evaluate(RUN / 'best_bottle.pt', 'selected-tta640', True)
    s.validate(m.MANIFEST['records'], m.DATA)
