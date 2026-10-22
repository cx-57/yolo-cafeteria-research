"""Verify a saved interrupted phase without rerunning training."""
from pathlib import Path
import gc
import importlib.util
import torch
from ultralytics import YOLO

RUN = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('bottle_phase', RUN / 'run.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
p.m.evaluate(RUN / 'best_bottle.pt', 'selected')
p.m.evaluate(RUN / 'best_bottle.pt', 'selected-tta640', True)
average = torch.load(RUN / 'averaged-state.pt', map_location='cpu', weights_only=True)
model = YOLO(str(p.SOURCE)).model.float().cpu()
model.load_state_dict(average['state'], strict=True)
model.eval()
weights = RUN / 'averaged-bottle.pt'
torch.save(dict(model=model.half(), ema=None, optimizer=None, epoch=-1,
                version=p.m.__version__, train_args=dict(task='detect')), weights)
p.s.save(RUN / 'averaging.json', dict(count=average['count'], first_epoch=average['first_epoch'],
         last_epoch=average['last_epoch'], weights_sha=p.s.sha(weights),
         method='Arithmetic mean of epoch EMA parameters with fixed BatchNorm running statistics'))
del average, model
gc.collect()
torch.mps.empty_cache()
p.m.evaluate(weights, 'averaged')
p.m.evaluate(weights, 'averaged-tta640', True)
p.s.validate(p.m.MANIFEST['records'], p.m.DATA)
