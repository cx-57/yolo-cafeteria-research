"""Train a fully trainable small-model control using the published WaDaBa notebook hyperparameters."""
from pathlib import Path
import gc
import importlib.util
import os
import shutil
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

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

class PaperTrainer(DetectionTrainer):
    def final_eval(self):
        pass  # Preserve optimizer state for any continuation.

if __name__ == '__main__':
    audit = dataset.verify()
    proof=s.load(RUN/'initialization.json')
    assert proof['architecture']=='YOLO11s' and s.sha(INITIAL)==proof['weights_sha']
    assert s.sha(Path(proof['source']))==proof['source_sha']
    if not (RUN / 'selection.json').exists():
        baseline = m.evaluate(INITIAL, 'baseline', batch=4)
        shutil.copy2(INITIAL, RUN / 'best_bottle.pt')
        s.save(RUN / 'selection.json', dict(epoch=0, AP50=baseline['AP50'], weights_sha=s.sha(INITIAL)))
    s.save(RUN / 'experiment.json', dict(target='bottle', hypothesis='Test full feature fine-tuning using the author notebook optimizer and schedule on our unchanged data',
               source=s.load(RUN / 'initialization.json'), data=str(DATA_CONFIG),
               source_dataset_fingerprint=m.MANIFEST['source_fingerprint'],
               validation_protocol=m.MANIFEST['validation_protocol'], validation_images=246,
               validation_bottle_objects=67, training_unique_images=audit['training_unique_images'], added_images=57, added_bottle_boxes=109, quality_holds=15, hard_negative_photos=12, hard_negative_repeats=1, external_review='AI visual review; not human verified',
               training_exposures_per_epoch=audit['training_exposures'], context_positive_repeats=8, external_positive_repeats=1,
               epochs=20, batch=4, imgsz=640, freeze=0, optimizer='AdamW',
               lr0=.000909, lrf=.001, momentum=.9, weight_decay=.0005, warmup_epochs=3., seed=0, nbs=64, mosaic=1., close_mosaic=10, scale=.5,
               normalization='Native training-mode BatchNorm running statistics; hardware-adapted microbatch 4', independent_test=False))
    resume = (RUN / 'last_resume.pt').exists()
    phase_start = s.load(RUN / 'progress.json')['epoch'] if resume else 0
    phase_epochs = int(os.environ.get('BOTTLE_PHASE_EPOCHS', '20'))

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
    def trainability(trainer):
        marker=(trainer.epoch,trainer.batch_size)
        if getattr(trainer,'paper_verified_epoch',None)==marker:return
        frozen=[name for name,p in trainer.model.named_parameters() if not p.requires_grad]
        assert frozen==['model.23.dfl.conv.weight'], frozen
        norms=[module for module in trainer.model.modules() if isinstance(module,torch.nn.BatchNorm2d)]
        assert norms and all(module.training for module in norms)
        s.save(RUN/'trainability.json',dict(epoch=trainer.epoch+1,frozen_parameters=frozen,all_feature_parameters_trainable=True,all_BatchNorm_modules_training=True,trainable_parameters=sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),normalization_modules=len(norms),physical_batch=trainer.batch_size))
        trainer.paper_verified_epoch=marker
    assert model.model.yaml['scale']=='s' and model.model.model[-1].nc==1
    model.add_callback('on_train_batch_start',trainability)
    model.add_callback('on_model_save', completed)
    model.train(trainer=PaperTrainer, resume=resume, data=str(DATA_CONFIG), classes=[0],
                epochs=20, batch=4, imgsz=640, device='mps', workers=0,
                optimizer='AdamW', lr0=.000909, lrf=.001, momentum=.9, weight_decay=.0005, warmup_bias_lr=0., warmup_epochs=3.,
                nbs=64, seed=0, patience=0, freeze=0, plots=False, project=str(RUN),
                name='train', save=True, mosaic=1., close_mosaic=10, scale=.5)
    del model
    gc.collect()
    torch.mps.empty_cache()
    m.evaluate(RUN / 'best_bottle.pt', 'selected', batch=4)
    m.evaluate(RUN / 'best_bottle.pt', 'selected-tta640', True)
    dataset.verify()
