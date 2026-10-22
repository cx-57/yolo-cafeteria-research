"""Recalibrate changed-layer BN buffers using training photos only."""
from pathlib import Path
import hashlib
import importlib.util
import random
import gc
import json
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO, __version__
from ultralytics.data.augment import LetterBox

RUN = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('bottle_common', RUN.parent / 'plastic-specialist-s-v2/run.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s = m.s
records = [r for r in m.MANIFEST['records'] if r['split'] == 'train']
random.Random(42).shuffle(records)
assert len(records) == 1056

def parameters_sha(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()

for tag, source, first in [
    ('head-only', RUN.parent / 'plastic-specialist-s-v3/initial.pt', 23),
    ('unfrozen', RUN.parent / 'plastic-specialist-s-v3/last_resume.pt', 10),
]:
    out = RUN / tag
    out.mkdir(exist_ok=True)
    m.RUN = out
    weights = out / 'recalibrated.pt'
    if not weights.exists():
        model = YOLO(str(source)).model.float().cpu().eval()
        before = parameters_sha(model)
        selected = [(name, module) for name, module in model.named_modules()
                    if isinstance(module, torch.nn.BatchNorm2d) and int(name.split('.')[1]) >= first]
        old_momenta = [module.momentum for _, module in selected]
        for _, module in selected:
            module.reset_running_stats()
            module.momentum = None
            module.train()
        model.to('mps')
        with torch.no_grad():
            for start in range(0, len(records), 8):
                images = []
                for record in records[start:start + 8]:
                    image = np.array(Image.open(m.DATA / 'images/train' / record['name']).convert('RGB'))
                    image = LetterBox(new_shape=(640, 640))(image=image)
                    images.append(torch.from_numpy(image.transpose(2, 0, 1).copy()))
                batch = torch.stack(images).to('mps').float() / 255
                model(batch)
                if start % 128 == 0:
                    print('CALIBRATION', tag, start + len(images), '/', len(records), flush=True)
        model.cpu().eval()
        for (_, module), momentum in zip(selected, old_momenta):
            module.momentum = momentum
        assert parameters_sha(model) == before, 'Calibration changed learned parameters'
        torch.save(dict(model=model.half(), ema=None, optimizer=None, epoch=-1,
                        version=__version__, train_args=dict(task='detect')), weights)
        s.save(out / 'calibration.json', dict(source=str(source), source_sha=s.sha(source),
                   weights_sha=s.sha(weights), training_images=1056, validation_images_used=0,
                   batch=8, batchnorm_layers=[name for name, _ in selected],
                   estimator='Cumulative mean of batch statistics over one shuffled pass',
                   learned_parameters_unchanged=True, learned_parameters_sha=before))
        del model, batch, images, selected
        gc.collect()
        torch.mps.empty_cache()
    report = m.evaluate(weights, 'verified')
    augmented = m.evaluate(weights, 'verified-tta640', True)
    if max(report['AP50'], augmented['AP50']) >= .85:
        break
s.validate(m.MANIFEST['records'], m.DATA)
