"""Compare frozen BatchNorm mismatch on fixed training photos; no val tuning."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

ROOT = Path.cwd()
DATA = Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-capture-validation')
records = json.loads((DATA / 'manifest.json').read_text())['records']
photos = [r for r in records if r['split'] == 'train' and r['rows'] and r['name'].startswith('trash_detection__')][:4]
photos += [r for r in records if r['split'] == 'train' and not r['rows'] and r['name'].startswith('plate__')][:4]
torch.set_num_threads(2)
report = dict(scope='training-only diagnostic; no validation images', photos=[r['name'] for r in photos], models={})
for tag, path in [('before', 'plastic-specialist-s-v3/initial.pt'), ('after', 'plastic-specialist-s-v3/last_resume.pt')]:
    model = YOLO(str(ROOT / 'model/bottle-focus' / path)).model.float().cpu().eval()
    samples = []
    def hook(module, inputs):
        x = inputs[0].float()
        mean, var = x.mean((0, 2, 3)), x.var((0, 2, 3), unbiased=False)
        z = ((mean - module.running_mean).abs() / (module.running_var + module.eps).sqrt()).mean()
        logvar = ((var + module.eps) / (module.running_var + module.eps)).log().abs().mean()
        samples.append((float(z), float(logvar)))
    handles = [module.register_forward_pre_hook(hook) for name, module in model.named_modules()
               if isinstance(module, torch.nn.BatchNorm2d) and int(name.split('.')[1]) >= 10]
    with torch.no_grad():
        for r in photos:
            image = np.array(Image.open(DATA / 'images/train' / r['name']).convert('RGB'))
            image = LetterBox(new_shape=(640, 640))(image=image)
            x = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().unsqueeze(0) / 255
            model(x)
    for handle in handles:
        handle.remove()
    report['models'][tag] = dict(checkpoint=str(path), mean_abs_channel_shift_in_running_std=float(np.mean([v[0] for v in samples])),
                                 mean_abs_log_variance_ratio=float(np.mean([v[1] for v in samples])), observations=len(samples))
    del model
out = Path(__file__).with_suffix('.json')
out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report['models']), flush=True)
