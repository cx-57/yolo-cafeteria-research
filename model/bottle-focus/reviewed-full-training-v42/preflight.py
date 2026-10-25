"""Freeze and check the full-training setup without training or model inference."""
from pathlib import Path
import datetime
import gc
import shutil
import torch
from ultralytics import YOLO,__version__
import build
import run
RUN=run.RUN;ROOT=run.ROOT;sha=build.sha;load=run.load;save=build.save
RUNTIME=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-capture-validation')
def main():
 assert not (RUN/'training-start.json').exists() and not (RUN/'authorization.json').exists(),'Do not alter an authorized/running declaration'
 audit=build.verify();code_names=[p.name for p in RUN.glob('*.py') if not p.name.startswith('test_')]
 # Include code dependencies of both prior frozen experiments, plus their receipts.
 references=['monitor-campaign-v34/monitor-initial.json','source-coverage-diagnostic-v40/balanced-litter.json',
             'monitor-campaign-v34/dataset.json','monitor-campaign-v34/data.yaml','source-coverage-diagnostic-v40/dataset.json','source-coverage-diagnostic-v40/data.yaml',
             'reviewed-neck-monitor-v36/declaration.json','source-coverage-diagnostic-v40/declaration.json']
 for folder in ['reviewed-neck-monitor-v36','source-coverage-diagnostic-v40']:
  prior=load(ROOT/folder/'declaration.json')
  references += [str((ROOT/folder/name).resolve().relative_to(ROOT.resolve())) for name in prior['code_shas']]
 references+=['source-coverage-diagnostic-v40/compile_sources.py','source-coverage-diagnostic-v40/admission.py']
 declaration=dict(initial_weights_sha=run.INITIAL_SHA,initial_weights=str(run.SOURCE),dataset_sha=sha(RUN/'dataset.json'),data_yaml_sha=sha(RUN/'data.yaml'),
                  code_shas={name:sha(RUN/name) for name in sorted(code_names)},reference_shas={name:sha(ROOT/name) for name in sorted(set(references))},recipe=run.RECIPE,
                  training_photos=audit['training_photos'],training_bottle_boxes=audit['training_bottle_boxes'],training_exposures=audit['training_exposures'],monitor_photos=68,monitor_boxes=35,
                  native_selection='combined monitor best AP50, best AP50-95 and latest; at most3distinct trained candidates',fresh_selection='maximum minimum of four source metric deltas, then mean delta; prefer initial then earliest',
                  benchmark_gate='all four source deltas nonnegative and mean gain>=.01',final_distinct_benchmark_candidates=1,
                  target_AP50=.9,aspirational_AP50_95=[.64,.70],promotion_floor=dict(AP50=.8407106817291071,AP50_95=.620605656065828),
                  full_training_paused=True,user_authorization_required=True,independent_NHS_test=False,ultralytics_version=__version__)
 save(RUN/'declaration.json',declaration);run.declaration_check()
 frozen=load(ROOT/'validation-review/frozen-files-verification.json')
 assert sha(RUNTIME/'manifest.json')==frozen['manifest_sha']
 for record in frozen['files']:
  assert sha(RUNTIME/'images/val'/record['name'])==record['image_sha']
  assert sha(RUNTIME/'labels/val'/(Path(record['name']).stem+'.txt'))==record['label_sha']
 assert sha(ROOT/'quality-head-blend-v19/alpha0.25-bottle.pt')==sha(run.SOURCE)==run.INITIAL_SHA
 legacy=sha(ROOT/'context-head-soup-v23/alpha0.5-bottle.pt');assert legacy=='2cd1efe2d7c187df05ed223ba2a18e147393358179f223c264663a2114812289'
 assert sha(ROOT/'verified-incumbent/bottle.pt')==legacy
 model=YOLO(str(run.SOURCE));assert model.names=={0:'bottle'} and model.model.yaml['scale']=='s'
 model.model.eval()
 for name,param in model.model.named_parameters():param.requires_grad_('.dfl.' not in name)
 proof=run.trainability(model.model)
 del model;gc.collect();torch.mps.empty_cache()
 free=shutil.disk_usage(RUN).free;assert free>=2*1024**3,'Need2GiBfree before launch'
 save(RUN/'preflight.json',dict(verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),declaration_sha=sha(RUN/'declaration.json'),dataset_sha=sha(RUN/'dataset.json'),
      training_photos=audit['training_photos'],training_bottle_boxes=audit['training_bottle_boxes'],training_exposures=audit['training_exposures'],permanent_monitor_photos_excluded=68,
      trainability_without_optimizer_steps=proof,frozen_benchmark_photo_and_label_hashes_verified=246,manifest_sha=frozen['manifest_sha'],balanced_weights_sha=run.INITIAL_SHA,legacy_weights_sha=legacy,
      free_disk_bytes=free,full_training_started=False,model_inference_performed=False,authorization_record_exists=False))
 print('FULL_TRAINING_READY_BUT_PAUSED',audit['training_photos'],audit['training_bottle_boxes'],audit['training_exposures'],free,flush=True)
if __name__=='__main__':main()
