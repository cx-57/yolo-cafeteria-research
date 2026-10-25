"""Require retention and source gains before at most one final benchmark assessment."""
from pathlib import Path
import data
import evaluate
import protocol
import run
RUN=data.RUN;ROOT=data.ROOT;load=data.load;save=data.save;sha=data.sha

def main():
 run.check_declaration();audit=data.verify();assert load(RUN/'training-complete.json')['declaration_sha']==sha(RUN/'declaration.json')
 cohorts=evaluate.cohorts(audit);retention=evaluate.evaluate(run.SOURCE,cohorts[0]);old=load(ROOT/'monitor-campaign-v34/monitor-initial.json');litter=load(ROOT/'source-coverage-diagnostic-v40/balanced-litter.json')
 assert old['weights_sha']==litter['weights_sha']==run.INITIAL_SHA
 reference=[retention['AP50'],retention['AP50_95'],old['AP50'],old['AP50_95'],litter['AP50'],litter['AP50_95']]
 candidates=[dict(role='initial',epoch=0,weights_sha=run.INITIAL_SHA,weights=str(run.SOURCE),values=reference)];seen={run.INITIAL_SHA}
 for role in ['ap50','localization','latest']:
  proof=load(RUN/(f'native-{role}.json' if role!='latest' else 'progress.json'))
  weights=run.WEIGHTS/f'best-{role}.pt' if role!='latest' else data.CACHE/'train/weights/last.pt';digest=sha(weights);assert digest==proof['weights_sha']
  if digest in seen:continue
  seen.add(digest);reports=[evaluate.evaluate(weights,c) for c in cohorts]
  candidates.append(dict(role=role,epoch=proof['epoch'],weights=reports[0]['weights'],weights_sha=digest,values=[v for report in reports for v in [report['AP50'],report['AP50_95']]],reports=reports))
 chosen=protocol.choose(candidates,reference);save(RUN/'source-monitor-selection.json',dict(reference=reference,candidates=candidates,chosen=chosen))
 result=dict(chosen=chosen,retention_and_source_gates_passed=chosen['role']!='initial',original_benchmark_used=False,declaration_sha=sha(RUN/'declaration.json'),independent_NHS_test=False)
 if chosen['role']!='initial':
  locks=data.parent.sources.module('v44_benchmark_lock',ROOT/'monitor-campaign-v34/selection.py');locks.bind_benchmark(RUN/'benchmark-attempt.json',chosen['weights_sha'],sha(RUN/'dataset.json'))
  common=data.parent.v36.m
  cohort=dict(name='original-benchmark',yaml=str(common.DATA/'data.yaml'),photos=246,boxes=67,fingerprint=common.MANIFEST['source_fingerprint'],validation_protocol=common.MANIFEST['validation_protocol'],dataset_sha=sha(common.DATA/'manifest.json'))
  report=evaluate.evaluate(Path(chosen['weights']),cohort);nonregress=report['AP50']>=.8407106817291071 and report['AP50_95']>=.620605656065828
  result.update(original_benchmark_used=True,benchmark_report=report,promotion_eligible=nonregress,move_on_eligible=nonregress and report['AP50']>=.9)
 save(RUN/'campaign-result.json',result);print('CLASSIFIER_OUTCOME',result,flush=True)
if __name__=='__main__':main()
