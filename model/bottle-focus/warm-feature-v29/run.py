"""Adapt all features from the protected specialist with a conservative learning rate and fixed BatchNorm."""
from pathlib import Path
import gc
import importlib.util
import os
import shutil
import subprocess
import sys
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


INITIAL_SOURCE = RUN.parent / 'context-head-soup-v23/alpha0.5-bottle.pt'
INITIAL_PROOF = RUN.parent / 'context-head-soup-v23/alpha0.5-ordinary640-iou0.5-cpu.json'
INITIAL_SHA = '2cd1efe2d7c187df05ed223ba2a18e147393358179f223c264663a2114812289'

def initialize():
    proof = s.load(INITIAL_PROOF)
    assert s.sha(INITIAL_SOURCE) == proof['weights_sha'] == INITIAL_SHA
    assert proof['source_dataset_fingerprint'] == m.MANIFEST['source_fingerprint']
    assert proof['validation_protocol'] == m.MANIFEST['validation_protocol']
    if not INITIAL.exists():
        shutil.copy2(INITIAL_SOURCE, INITIAL)
        s.save(RUN / 'initialization.json', dict(source=str(INITIAL_SOURCE),
                   weights_sha=INITIAL_SHA, source_verification=proof,
                   reason='Pinned immutable 84.0784% specialist checkpoint; separate global incumbent protected.'))
    assert s.sha(INITIAL) == INITIAL_SHA
    ledger = s.load(RUN / 'initialization.json')
    assert ledger['weights_sha'] == INITIAL_SHA
    assert Path(ledger['source']).resolve() == INITIAL_SOURCE.resolve()
    assert ledger['source_verification']['weights_sha'] == INITIAL_SHA

if __name__ == '__main__':
    audit = dataset.verify()
    initialize()
    if not (RUN / 'selection.json').exists():
        baseline = m.evaluate(INITIAL, 'baseline')
        shutil.copyfile(INITIAL, RUN / 'best_bottle.pt')
        s.save(RUN / 'selection.json', dict(epoch=0, AP50=baseline['AP50'], weights_sha=s.sha(INITIAL)))
    s.save(RUN / 'experiment.json', dict(target='bottle', hypothesis='Gentle full-feature adaptation from the protected specialist improves the reviewed-data result without resetting normalization',
               source=s.load(RUN / 'initialization.json'), data=str(DATA_CONFIG),
               source_dataset_fingerprint=m.MANIFEST['source_fingerprint'],
               validation_protocol=m.MANIFEST['validation_protocol'], validation_images=246,
               validation_bottle_objects=67, training_unique_images=audit['training_unique_images'], added_images=74, added_bottle_boxes=129, round4_added_images=17, round4_added_boxes=20, quality_holds=15, hard_negative_photos=12, hard_negative_repeats=1, external_review='AI visual review; not human verified',
               training_exposures_per_epoch=audit['training_exposures'], context_positive_repeats=8, external_positive_repeats=1, round4_positive_repeats=4,
               epochs=40, batch=4, imgsz=640, seed=43, freeze=0, optimizer='AdamW',
               lr0=.000001, nbs=16, mosaic=.5, close_mosaic=0, scale=.5,
               normalization='All pretrained BatchNorm running statistics fixed', independent_test=False))
    resume = (RUN / 'last_resume.pt').exists()
    phase_start = s.load(RUN / 'progress.json')['epoch'] if resume else 0
    phase_epochs = int(os.environ.get('BOTTLE_PHASE_EPOCHS', '3'))

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
        if getattr(trainer,'v29_checked_epoch',None)==trainer.epoch:return
        frozen=[name for name,p in trainer.model.named_parameters() if not p.requires_grad]
        assert frozen==['model.23.dfl.conv.weight'], frozen
        norms=[module for module in trainer.model.modules() if isinstance(module,torch.nn.BatchNorm2d)]
        assert norms and not any(module.training for module in norms)
        s.save(RUN/'trainability.json',dict(epoch=trainer.epoch+1,all_feature_parameters_trainable=True,all_BatchNorm_running_statistics_fixed=True,trainable_parameters=sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),normalization_modules=len(norms)))
        trainer.v29_checked_epoch=trainer.epoch
    assert model.model.yaml['scale']=='s' and model.model.model[-1].nc==1
    model.add_callback('on_train_batch_start',trainability)
    model.add_callback('on_model_save', completed)
    model.train(trainer=m.Trainer, resume=resume, data=str(DATA_CONFIG), classes=[0],
                epochs=40, batch=4, imgsz=640, device='mps', workers=0,
                optimizer='AdamW', lr0=.000001, warmup_bias_lr=0., warmup_epochs=1.,
                nbs=16, seed=43, patience=0, freeze=0, plots=False, project=str(RUN),
                name='train', save=True, mosaic=.5, close_mosaic=0, scale=.5)
    del model
    gc.collect()
    torch.mps.empty_cache()
    m.evaluate(RUN / 'best_bottle.pt', 'selected')
    m.evaluate(RUN / 'best_bottle.pt', 'selected-tta640', True)
    dataset.verify()
    subprocess.run([sys.executable,str(RUN.parent/'external-data-v7/validate_exact.py'),'--run',str(RUN),'--candidate','trained','--ordinary'],check=True)
