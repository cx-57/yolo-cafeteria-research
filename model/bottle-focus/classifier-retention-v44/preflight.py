"""Freeze the authorized bounded classifier experiment and verify protected assets."""
from pathlib import Path
import datetime
import gc
import shutil
import torch
from ultralytics import YOLO,__version__
import data
import geometry
import run
RUN=data.RUN;ROOT=data.ROOT;sha=data.sha;load=data.load;save=data.save

def main():
 assert not (RUN/'measurement-start.json').exists() and not (RUN/'training-start.json').exists()
 audit=data.verify();review=load(RUN/'retention-review.json')
 assert review['dataset_sha']==sha(RUN/'dataset.json') and review['retention_fingerprint']==audit['retention_fingerprint'] and review['independent_second_review'] is True
 assert review['human_verified'] is False and review['full_scene_reviewed'] is True
 assert review['records']==[dict(id=r['id'],image_sha=r['image_sha'],rows=r['rows'],accepted=True,verified_absence=not r['rows']) for r in audit['retention_records']]
 assert sha(run.SOURCE)==run.INITIAL_SHA and sha(run.V42)=='95370c9b321b57feebbf11204b51a7a8393085dd80bb0ae483b0a1a1f2433013'
 references=['reviewed-full-training-v42/declaration.json','monitor-campaign-v34/monitor-initial.json','source-coverage-diagnostic-v40/balanced-litter.json',
             'monitor-campaign-v34/data.yaml','source-coverage-diagnostic-v40/data.yaml','monitor-campaign-v34/dataset.json','source-coverage-diagnostic-v40/dataset.json',
             'optimization-diagnostic-v43/training-first-review.json','optimization-diagnostic-v43/training-second-review.json','optimization-diagnostic-v43/trash-training-review-records.json']
 prior=load(ROOT/'reviewed-full-training-v42/declaration.json')
 references += ['reviewed-full-training-v42/'+name for name in prior['code_shas']]+list(prior['reference_shas'])
 declaration=dict(recipe=run.RECIPE,initial_weights_sha=run.INITIAL_SHA,v42_weights_sha=sha(run.V42),dataset_sha=sha(RUN/'dataset.json'),data_yaml_sha=sha(RUN/'data.yaml'),retention_review_sha=sha(RUN/'retention-review.json'),
                  code_shas={p.name:sha(p) for p in sorted(RUN.glob('*.py')) if not p.name.startswith('test_')},reference_shas={name:sha(ROOT/name) for name in sorted(set(references))},
                  classifier_parameter_prefix=geometry.PREFIX,fixed_nonclassifier_parameters_and_all_buffers=True,EMA_fixed_tensor_restoration=True,
                  training_photos=audit['training_photos'],training_boxes=audit['training_boxes'],training_exposures=audit['training_exposures'],retention_photos=24,retention_boxes=16,retention_fingerprint=audit['retention_fingerprint'],
                  previously_seen_retention=True,independent_NHS_test=False,source_monitor_baselines='initial pinned exact checkpoint receipts; retention measured before optimizer steps',
                  maximum_distinct_trained_candidates=3,final_distinct_benchmark_candidates=1,retention_gate_tolerance=.005,minimum_other_source_mean_gain=.01,other_source_each_nonregression=True,
                  early_stop_after_epoch2_if_either_retention_drop_exceeds=.03,terminal_state_recovered_before_any_training_on_restart=True,
                  target_AP50=.9,promotion_floor=dict(AP50=.8407106817291071,AP50_95=.620605656065828),ultralytics_version=__version__,user_authorized_max_epochs=20)
 save(RUN/'declaration.json',declaration);run.check_declaration()
 frozen=load(ROOT/'validation-review/frozen-files-verification.json');benchmark=data.parent.v36.m.DATA
 assert sha(benchmark/'manifest.json')==frozen['manifest_sha']
 for record in frozen['files']:
  assert sha(benchmark/'images/val'/record['name'])==record['image_sha']
  assert sha(benchmark/'labels/val'/(Path(record['name']).stem+'.txt'))==record['label_sha']
 assert sha(ROOT/'quality-head-blend-v19/alpha0.25-bottle.pt')==run.INITIAL_SHA
 assert sha(ROOT/'verified-incumbent/bottle.pt')=='2cd1efe2d7c187df05ed223ba2a18e147393358179f223c264663a2114812289'
 model=YOLO(str(run.SOURCE)).model.float().cpu().eval();names=geometry.select(model);reference=geometry.fixed_state(model);fingerprint=geometry.fingerprint(reference);del model;gc.collect()
 free=shutil.disk_usage(RUN).free;assert free>=2*1024**3
 save(RUN/'preflight.json',dict(verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),declaration_sha=sha(RUN/'declaration.json'),training_photos=audit['training_photos'],training_boxes=audit['training_boxes'],training_exposures=audit['training_exposures'],retention_photos=24,retention_boxes=16,
      trainable_classifier_tensors=len(names),fixed_tensor_fingerprint=fingerprint,frozen_benchmark_photo_and_label_hashes_verified=246,initial_weights_sha=run.INITIAL_SHA,free_disk_bytes=free,real_experiment_training_started=False,
      synthetic_trainer_smoke_passed=True,synthetic_smoke_log_sha=sha(Path('/tmp/v44-trainer-smoke.log')),independent_review='No remaining concrete P1/P2 after terminal-recovery fix; all retention photos reviewed'))
 save(RUN/'authorization.json',dict(declaration_sha=sha(RUN/'declaration.json'),optimization_authorized=True,maximum_epochs_authorized=20,experiment_epochs=6,user_instruction='tihs is ASS bro, optimize tis',scope='Bounded bottle optimization under user20epoch cap; preserve better incumbent and unchanged benchmark'))
 print('CLASSIFIER_PREFLIGHT_READY',audit['training_photos'],audit['training_boxes'],audit['training_exposures'],fingerprint,flush=True)
if __name__=='__main__':main()
