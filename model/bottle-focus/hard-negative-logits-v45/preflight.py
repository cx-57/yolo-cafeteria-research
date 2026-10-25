"""Freeze the scope-consistent three-epoch hard-negative logit calibration."""
from pathlib import Path
import datetime
import gc
import shutil
from ultralytics import YOLO,__version__
import data
import geometry
import run
RUN=data.RUN;ROOT=data.ROOT;sha=data.sha;load=data.load;save=data.save

def main():
 assert not (RUN/'measurement-start.json').exists() and not (RUN/'training-start.json').exists()
 audit=data.verify();scope=load(RUN/'user-scope.json');assert scope['target']=='disposable_plastic_drink_bottles'
 old_review=load(data.V44/'retention-review.json');review=dict(old_review,dataset_sha=sha(RUN/'dataset.json'),parent_review_sha=sha(data.V44/'retention-review.json'),parent_dataset_sha=sha(data.V44/'dataset.json'))
 assert review['retention_fingerprint']==audit['retention_fingerprint'] and review['records']==[dict(id=r['id'],image_sha=r['image_sha'],rows=r['rows'],accepted=True,verified_absence=not r['rows']) for r in audit['retention_records']]
 save(RUN/'retention-review.json',review)
 references=['classifier-retention-v44/declaration.json','classifier-retention-v44/dataset.json','classifier-retention-v44/source-selection.json','classifier-retention-v44/retention-review.json',
             'monitor-campaign-v34/monitor-initial.json','source-coverage-diagnostic-v40/balanced-litter.json','monitor-campaign-v34/data.yaml','source-coverage-diagnostic-v40/data.yaml','monitor-campaign-v34/dataset.json','source-coverage-diagnostic-v40/dataset.json']
 prior=load(data.V44/'declaration.json');references += ['classifier-retention-v44/'+n for n in prior['code_shas']]+list(prior['reference_shas'])
 assert sha(run.SOURCE)==run.INITIAL_SHA
 model=YOLO(str(run.SOURCE)).model.float().cpu().eval();names=geometry.select(model);reference=geometry.fixed_state(model);params=sum(p.numel() for n,p in model.named_parameters() if n in names)
 assert len(names)==6 and params==387;fingerprint=geometry.fingerprint(reference);del model;gc.collect()
 declaration=dict(recipe=run.RECIPE,initial_weights_sha=run.INITIAL_SHA,dataset_sha=sha(RUN/'dataset.json'),data_yaml_sha=sha(RUN/'data.yaml'),target_scope=scope['target'],user_scope_sha=sha(RUN/'user-scope.json'),
                  retention_review_sha=sha(RUN/'retention-review.json'),hard_review_sha=sha(RUN/'hard-review.json'),code_shas={p.name:sha(p) for p in sorted(RUN.glob('*.py')) if not p.name.startswith('test_')},reference_shas={n:sha(ROOT/n) for n in sorted(set(references))},
                  allowed_parameter_names=sorted(names),mutable_parameter_tensors=6,mutable_parameters=387,all_spatial_and_classifier_features_fixed=True,all_BatchNorm_buffers_fixed=True,EMA_fixed_tensor_restoration=True,
                  training_photos=audit['training_photos'],training_boxes=audit['training_boxes'],training_exposures=audit['training_exposures'],retention_photos=24,retention_boxes=16,retention_fingerprint=audit['retention_fingerprint'],
                  user_authorized_max_epochs=20,maximum_distinct_trained_candidates=3,final_distinct_benchmark_candidates=1,retention_gate_tolerance=.005,minimum_other_source_mean_gain=.01,other_source_each_nonregression=True,
                  early_stop_after_epoch2_if_either_retention_drop_exceeds=.03,terminal_state_recovered_before_any_training_on_restart=True,
                  target_AP50=.9,promotion_floor=dict(AP50=.8407106817291071,AP50_95=.620605656065828),independent_NHS_test=False,retention_previously_seen_by_initial=True,ultralytics_version=__version__)
 save(RUN/'declaration.json',declaration);run.check_declaration()
 frozen=load(ROOT/'validation-review/frozen-files-verification.json');benchmark=data.parent.v36.m.DATA
 assert sha(benchmark/'manifest.json')==frozen['manifest_sha']
 for record in frozen['files']:
  assert sha(benchmark/'images/val'/record['name'])==record['image_sha']
  assert sha(benchmark/'labels/val'/(Path(record['name']).stem+'.txt'))==record['label_sha']
 assert sha(ROOT/'quality-head-blend-v19/alpha0.25-bottle.pt')==run.INITIAL_SHA
 assert sha(ROOT/'verified-incumbent/bottle.pt')=='2cd1efe2d7c187df05ed223ba2a18e147393358179f223c264663a2114812289'
 free=shutil.disk_usage(RUN).free;assert free>=2*1024**3
 save(RUN/'preflight.json',dict(verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),declaration_sha=sha(RUN/'declaration.json'),training_photos=audit['training_photos'],training_boxes=audit['training_boxes'],training_exposures=audit['training_exposures'],
      retention_photos=24,retention_boxes=16,mutable_parameters=387,fixed_tensor_fingerprint=fingerprint,frozen_benchmark_image_label_hashes_verified=246,initial_weights_sha=run.INITIAL_SHA,free_disk_bytes=free,
      training_started=False,synthetic_trainer_smoke_passed=True,focused_tests_log_sha=sha(Path('/tmp/v45-all-tests.log'))))
 save(RUN/'authorization.json',dict(declaration_sha=sha(RUN/'declaration.json'),optimization_authorized=True,maximum_epochs_authorized=20,experiment_epochs=3,user_instruction='Optimize bottle after regression; user explicitly confirmed only disposable plastic drink bottles',scope='Three-epoch hard-negative calibration under20epoch cap; unchanged original benchmark'))
 print('HARD_NEGATIVE_LOGITS_READY',audit['training_photos'],audit['training_boxes'],audit['training_exposures'],params,fingerprint,flush=True)
if __name__=='__main__':main()
