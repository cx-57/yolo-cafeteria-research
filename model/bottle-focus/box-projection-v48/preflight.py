"""Freeze box-projection experiment after real protection/data checks."""
import gc
import shutil
from pathlib import Path
from ultralytics import YOLO,__version__
import data
import run
import geometry
RUN=data.RUN;ROOT=data.ROOT;sha=data.sha;load=data.load;save=data.save

def main():
 assert not (RUN/'measurement-start.json').exists();audit=data.verify()
 prior=data.prior;prior_decl=load(data.V45/'declaration.json')
 assert prior_decl['code_shas']=={n:sha(data.V45/n) for n in prior_decl['code_shas']}
 assert prior_decl['reference_shas']=={n:sha(ROOT/n) for n in prior_decl['reference_shas']}
 references=['hard-negative-logits-v45/'+n for n in prior_decl['code_shas']]+list(prior_decl['reference_shas'])
 references+=['hard-negative-logits-v45/declaration.json','hard-negative-logits-v45/dataset.json','hard-negative-logits-v45/user-scope.json','hard-negative-logits-v45/retention-review.json','classifier-retention-v44/source-selection.json','openimages-drink-staging-v47/screen.py','openimages-drink-staging-v47/stage.py','validation-review/frozen-files-verification.json']
 references += [str(Path(p).relative_to(ROOT)) for p in audit['source_input_shas']]
 model=YOLO(str(run.SOURCE)).model.float().cpu().eval();names=geometry.select(model);params=sum(p.numel() for n,p in model.named_parameters() if n in names);assert len(names)==6 and params==12480
 fingerprint=geometry.fingerprint(geometry.fixed_state(model));del model;gc.collect()
 review=dict(load(data.V45/'retention-review.json'),dataset_sha=sha(RUN/'dataset.json'),parent_review_sha=sha(data.V45/'retention-review.json'),parent_dataset_sha=sha(data.V45/'dataset.json'));save(RUN/'retention-review.json',review)
 d=dict(recipe=run.RECIPE,initial_weights_sha=run.INITIAL_SHA,dataset_sha=sha(RUN/'dataset.json'),data_yaml_sha=sha(RUN/'data.yaml'),target_scope='disposable_plastic_drink_bottles',user_scope_sha=sha(RUN/'user-scope.json'),retention_review_sha=sha(RUN/'retention-review.json'),code_shas={p.name:sha(p) for p in sorted(RUN.glob('*.py')) if not p.name.startswith('test_')},reference_shas={n:sha(ROOT/n) for n in sorted(set(references))},allowed_parameter_names=sorted(names),mutable_parameters=params,mutable_parameter_tensors=6,classifier_and_intermediate_regression_features_fixed=True,all_BatchNorm_buffers_fixed=True,fixed_tensor_fingerprint=fingerprint,maximum_distinct_trained_candidates=3,maximum_original_benchmark_candidates=1,user_authorized_max_epochs=20,retention_gate_tolerance=.005,minimum_other_source_mean_gain=.01,other_source_each_nonregression=True,target_AP50=.9,promotion_floor=dict(AP50=.8407106817291071,AP50_95=.620605656065828),independent_NHS_test=False,ultralytics_version=__version__)
 assert sha(run.SOURCE)==run.INITIAL_SHA;save(RUN/'declaration.json',d);run.check_declaration()
 frozen=load(ROOT/'validation-review/frozen-files-verification.json');benchmark=data.parent.v36.m.DATA;assert sha(benchmark/'manifest.json')==frozen['manifest_sha']
 for r in frozen['files']:
  assert sha(benchmark/'images/val'/r['name'])==r['image_sha']
  assert sha(benchmark/'labels/val'/(Path(r['name']).stem+'.txt'))==r['label_sha']
 assert shutil.disk_usage(RUN).free>=2*1024**3
 save(RUN/'authorization.json',dict(declaration_sha=sha(RUN/'declaration.json'),optimization_authorized=True,maximum_epochs_authorized=20,experiment_epochs=3,user_instruction='Keep going with bottle optimization; only disposable plastic drink bottles'))
 save(RUN/'preflight.json',dict(declaration_sha=sha(RUN/'declaration.json'),training_photos=audit['training_photos'],training_boxes=audit['training_boxes'],training_exposures=audit['training_exposures'],new_photos=len(audit['added']),mutable_parameters=params,fixed_tensor_fingerprint=fingerprint,frozen_benchmark_image_label_hashes_verified=246,training_started=False))
 print('BOX_PROJECTION_PREFLIGHT_READY',len(audit['added']),params,fingerprint,flush=True)
if __name__=='__main__':main()
