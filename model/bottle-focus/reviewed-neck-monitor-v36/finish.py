"""Choose on monitor only, bind one distinct final benchmark candidate across restarts."""
from pathlib import Path
import importlib.util
import json

RUN=Path(__file__).resolve().parent
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
selection=module('v34_finish_selection',RUN/'selection.py')
shared=module('v34_benchmark_policy',RUN.parent/'joint-metric-v30/policy.py')
benchmark=module('v34_benchmark_eval',RUN.parent/'joint-metric-v30/evaluate.py')
monitor_eval=module('v34_monitor_eval',RUN/'monitor_evaluate.py')
receipt=module('v34_benchmark_receipt',RUN/'benchmark_receipt.py')
gate=module('v36_monitor_gate',RUN/'gate.py')

def finish(common,dataset,weights_dir):
 s=common.s;audit=dataset.verify();progress=s.load(RUN/'monitor-progress.json');assert selection.budget_finished(progress['epoch'])
 if (RUN/'campaign-result.json').exists():
  old=s.load(RUN/'campaign-result.json');assert old['training_dataset_audit_sha']==s.sha(RUN/'dataset.json')
  if old.get('benchmark_report'):
   reports=shared.verified_records(RUN);assert any(r['weights_sha']==old['chosen_weights_sha'] for r in reports)
  return old
 candidates=[('initial',weights_dir/'initial.pt',0)]
 for role in ('ap50','localization'):
  proof=s.load(RUN/f'native-{role}-selection.json');candidates.append((role,weights_dir/f'best-monitor-{role}.pt',proof['epoch']))
 candidates.append(('latest',weights_dir/'last_resume.pt',progress['epoch']))
 reports=[];seen=set()
 for role,path,epoch in candidates:
  digest=s.sha(path)
  if digest in seen:continue
  seen.add(digest);reports.append(monitor_eval.evaluate(RUN,path,role,epoch,common,audit,weights_dir))
 dataset.verify();chosen=selection.choose(reports,audit['monitor_fingerprint']);s.save(RUN/'monitor-reports.json',reports);s.save(RUN/'monitor-choice.json',chosen)
 assert s.sha(Path(chosen['weights']))==chosen['weights_sha']
 if chosen['role']=='initial' or not gate.eligible(chosen):
  reference=RUN.parent/'joint-metric-v30/phase0-initial.json';report=s.load(reference)
  assert report['weights_sha']==s.sha(weights_dir/'initial.pt') and report['source_dataset_fingerprint']==common.MANIFEST['source_fingerprint'] and report['validation_protocol']==common.MANIFEST['validation_protocol']
  result=dict(status='monitor_gate_not_met_benchmark_skipped',benchmark_report=None,benchmark_reference=str(reference),benchmark_reference_weights_sha=report['weights_sha'],benchmark_reference_AP50=report['AP50'],benchmark_reference_AP50_95=report['AP50_95'],AP50=None,AP50_95=None,new_benchmark_measurement=False,chosen_monitor_candidate_benchmarked=False)
 else:
  selection.bind_benchmark(RUN/'benchmark-attempt.json',chosen['weights_sha'],s.sha(RUN/'dataset.json'))
  report_path=RUN/'phase9-monitor-selected.json'
  if report_path.exists():
   report=receipt.reconcile(RUN,report_path,Path(chosen['weights']),common.MANIFEST,shared)
  else:report=benchmark.evaluate(RUN,Path(chosen['weights']),'monitor-selected',9,common,dataset,shared)
  receipt.promote(RUN,report_path,report,common.DATA.parents[1]/'weights/bottle-best-verified.pt')
  result=dict(status='joint_target_verified' if shared.target_met(report) else 'campaign_complete_below_joint_target',benchmark_report=str(report_path),AP50=report['AP50'],AP50_95=report['AP50_95'],new_benchmark_measurement=True)
 result.update(chosen_weights_sha=chosen['weights_sha'],chosen_monitor_role=chosen['role'],chosen_epoch=chosen['selected_epoch'],training_dataset_audit_sha=s.sha(RUN/'dataset.json'),monitor_fingerprint=audit['monitor_fingerprint'],selection_rule='maximum monitor AP50+AP50-95; ties prefer initial then earliest',monitor_gate=dict(AP50=gate.MIN_AP50,AP50_95=gate.MIN_AP95),full_training_started=False,independent_NHS_test=False)
 s.save(RUN/'campaign-result.json',result);print('FINAL_FIXED_BENCHMARK_OUTCOME',json.dumps(result),flush=True);return result
