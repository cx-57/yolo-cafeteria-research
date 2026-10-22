"""Train and verify a COCO-initialized, single-class bottle specialist."""
from pathlib import Path
import copy
import gc
import json
import shutil
import torch
from ultralytics import YOLO, __version__
from ultralytics.models.yolo.detect import DetectionTrainer
import utensil_packet as s

ROOT = Path.cwd()
RUN = Path(__file__).resolve().parent
DATA = Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-capture-validation')
MANIFEST = s.load(DATA / 'manifest.json')
s.NAMES = ['bottle']
s.validate_configuration(DATA, MANIFEST['records'])
s.validate(MANIFEST['records'], DATA)
assert MANIFEST['validation_images'] == 246 and MANIFEST['validation_bottle_objects'] == 67
torch.set_num_threads(2)

class Trainer(DetectionTrainer):
    def _model_train(self):
        super()._model_train()
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

    def final_eval(self):
        pass  # Keep optimizer state for continuation.

def evaluate(weights, tag, augment=False, batch=4):
    model = YOLO(str(weights))
    metrics = model.val(data=str(DATA / 'data.yaml'), classes=[0], imgsz=640,
                        batch=1 if augment else batch, device='mps', workers=0,
                        augment=augment, save_json=True, plots=False,
                        project=str(RUN / 'evaluations'), name=tag)
    assert list(metrics.box.ap_class_index) == [0] and int(metrics.nt_per_class[0]) == 67
    p, r, ap, ap95 = metrics.box.class_result(0)
    report = dict(weights=str(weights), weights_sha=s.sha(weights), AP50=float(ap),
                  AP50_95=float(ap95), precision=float(p), recall=float(r),
                  imgsz=640, batch=1 if augment else batch, nms_iou=.7,
                  test_time_augmentation=augment, validation_images=246,
                  bottle_objects=67, source_dataset_fingerprint=MANIFEST['source_fingerprint'],
                  validation_protocol=MANIFEST['validation_protocol'], independent_test=False)
    s.save(RUN / (tag + '.json'), report)
    print('BOTTLE_RESULT', tag, json.dumps(report), flush=True)
    del model
    gc.collect()
    torch.mps.empty_cache()
    return report

if __name__ == '__main__':
    initial = RUN / 'initial.pt'
    if not initial.exists():
        source = ROOT / 'model/weights/yolo11s.pt'
        original = YOLO(str(source)).model.float().cpu().eval()
        assert original.names[39] == 'bottle'
        projected = copy.deepcopy(original)
        head = projected.model[-1]
        for branch in head.cv3:
            old = branch[-1]
            new = torch.nn.Conv2d(old.in_channels, 1, old.kernel_size, old.stride,
                                  old.padding, bias=old.bias is not None)
            with torch.no_grad():
                new.weight.copy_(old.weight[39:40])
                if old.bias is not None:
                    new.bias.copy_(old.bias[39:40])
            branch[-1] = new
        head.nc = 1
        head.no = 4 * head.reg_max + 1
        projected.yaml['nc'] = 1
        projected.names = {0: 'bottle'}
        projected.criterion = None
        with torch.no_grad():
            x = torch.rand(1, 3, 96, 96)
            a, b = original(x)[0], projected(x)[0]
            torch.testing.assert_close(a[:, :4], b[:, :4], rtol=0, atol=0)
            torch.testing.assert_close(a[:, 43:44], b[:, 4:], rtol=1e-6, atol=1e-7)
        torch.save(dict(model=projected.half(), ema=None, optimizer=None, epoch=-1,
                        version=__version__, train_args=dict(task='detect')), initial)
        s.save(RUN / 'initialization.json', dict(source=str(source), source_sha=s.sha(source),
                   weights_sha=s.sha(initial), COCO_class=39, decoded_output_preserved=True))
        del original, projected, head, a, b, x
        gc.collect()
    if not (RUN / 'selection.json').exists():
        baseline = evaluate(initial, 'baseline')
        shutil.copy2(initial, RUN / 'best_bottle.pt')
        s.save(RUN / 'selection.json', dict(epoch=0, AP50=baseline['AP50']))

    def completed(trainer):
        score = float(trainer.validator.metrics.box.ap50[0])
        print('BOTTLE_EPOCH', trainer.epoch + 1, 'AP50', score, flush=True)
        shutil.copy2(trainer.last, RUN / 'last_resume.pt')
        if score > s.load(RUN / 'selection.json')['AP50']:
            shutil.copy2(trainer.last, RUN / 'best_bottle.pt')
            s.save(RUN / 'selection.json', dict(epoch=trainer.epoch + 1, AP50=score,
                                               weights_sha=s.sha(RUN / 'best_bottle.pt')))
        if score >= .85:
            trainer.stop = True

    resume = (RUN / 'last_resume.pt').exists()
    model = YOLO(str(RUN / 'last_resume.pt' if resume else initial))
    model.add_callback('on_model_save', completed)
    model.train(trainer=Trainer, resume=resume, data=str(DATA / 'data.yaml'), classes=[0],
                epochs=40, batch=4, imgsz=640, device='mps', workers=0,
                optimizer='AdamW', lr0=.0001, warmup_bias_lr=0., warmup_epochs=1.,
                nbs=16, seed=42, patience=0, freeze=23, plots=False, project=str(RUN),
                name='train', save=True, mosaic=.5, close_mosaic=10)
    del model
    gc.collect()
    torch.mps.empty_cache()
    evaluate(RUN / 'best_bottle.pt', 'selected')
    evaluate(RUN / 'best_bottle.pt', 'selected-tta640', True)
    s.validate(MANIFEST['records'], DATA)
