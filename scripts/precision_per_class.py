"""Validation-selected class-specific confidence thresholds for frozen YOLO specialists.

Exploratory 7-class OBJECT precision/recall only. This does not increase mAP or
prove correct disposal-bin assignments. Never optimize on the exposed test.
"""
import argparse
import json
import math
from pathlib import Path

import precision_sweep as p
import research_audit
import specialists as s
from storage import EVALUATION, ROOT

CLASS_NAMES=tuple(s.NAMES)

def class_options(rows):
    """Best true positives for each attainable false-positive count, per class."""
    answer={}
    for name in CLASS_NAMES:
        distinct={}
        for row in rows:
            score=row['per_class'][name]
            fp,tp=score['fp'],score['tp']
            candidate=dict(threshold=row['threshold'],tp=tp,fp=fp)
            old=distinct.get(fp)
            if old is None or tp>old['tp'] or (tp==old['tp'] and candidate['threshold']<old['threshold']):
                distinct[fp]=candidate
        answer[name]=sorted(distinct.values(),key=lambda item:(item['fp'],-item['tp']))
    return answer

def optimize(rows,goal=.97,min_recall=.60,min_predictions=100):
    if not (math.isfinite(goal) and 0<goal<=1 and
            math.isfinite(min_recall) and 0<=min_recall<=1 and
            type(min_predictions) is int and min_predictions>=1):
        raise ValueError('Invalid optimization target or guards')
    if not rows:
        raise ValueError('No threshold rows')
    options=class_options(rows)
    objects=sum(rows[0]['per_class'][name]['objects'] for name in CLASS_NAMES)
    # DP: among states with EXACTLY the same FP count, keep maximum TP.
    # With this objective, a lesser-TP state can never dominate at a later step.
    states={0:(0,())}
    for name in CLASS_NAMES:
        new={}
        for old_fp,(old_tp,old_choice) in states.items():
            for candidate in options[name]:
                fp=old_fp+candidate['fp']
                tp=old_tp+candidate['tp']
                choice=(*old_choice,(name,candidate['threshold']))
                old=new.get(fp)
                if old is None or tp>old[0] or (tp==old[0] and
                        sum(t for _,t in choice)<sum(t for _,t in old[1])):
                    new[fp]=(tp,choice)
        states=new

    valid=[]
    for fp,(tp,choice) in states.items():
        predictions=tp+fp
        precision=tp/predictions if predictions else None
        recall=tp/objects if objects else None
        if (precision is None or recall is None or recall<min_recall
                or predictions<min_predictions):
            continue
        valid.append(dict(precision=precision,recall=recall,tp=tp,fp=fp,
                          fn=objects-tp,predictions=predictions,objects=objects,
                          thresholds=dict(choice)))
    if not valid:
        return dict(selected=None,best_guarded=None)
    eligible=[x for x in valid if x['precision']>=goal-1e-12]
    # Optimize number of correctly detected objects, then precision.
    selected=max(eligible,key=lambda x:(x['tp'],x['precision'],-sum(x['thresholds'].values()))) if eligible else None
    best=max(valid,key=lambda x:(x['precision'],x['tp'],-sum(x['thresholds'].values())))
    return dict(selected=selected,best_guarded=best)

def evaluate_thresholds(records,predictions,thresholds):
    """Remove predictions below their class-specific thresholds, then match once."""
    if set(thresholds)!=set(CLASS_NAMES) or any(
            type(t) not in (int,float) or not math.isfinite(t) or not .5<=t<=1
            for t in thresholds.values()):
        raise ValueError('Missing or invalid class thresholds')
    filtered={name:[r for r in rows if r[4]>=thresholds[CLASS_NAMES[int(r[5])]]]
              for name,rows in predictions.items()}
    return s.recognition_at_threshold(records,filtered,confidence=.5)['micro']

def run(val_manifest,test_manifest,val_receipt,test_receipt,
        out,device,target,min_recall,min_predictions,skip_test):
    if out.exists():
        raise FileExistsError('Refusing to overwrite '+str(out))
    freeze=s.read(ROOT/'configs/research-freeze.json')
    research_audit.verify_freeze(freeze)
    config=s.read(s.CONFIG)
    for spec in config['models']:
        if s.sha(ROOT/spec['weights'])!=spec['sha256']:
            raise ValueError('Changed model checkpoint: '+spec['id'])

    print('Reading/generating frozen validation predictions for classwise thresholds...',flush=True)
    v_records,v_preds,v_hash=p.load_cohort(val_manifest,val_receipt,device,config)
    rows=[]
    for threshold in p.thresholds():
        scored=s.recognition_at_threshold(v_records,v_preds,confidence=threshold)
        rows.append(dict(threshold=threshold,per_class=scored['per_class'],
                         micro=scored['micro']))
    selection=optimize(rows,goal=target,min_recall=min_recall,min_predictions=min_predictions)
    selected=selection['selected']
    reference=evaluate_thresholds(v_records,v_preds,dict.fromkeys(CLASS_NAMES,.5))
    if selected:
        measured=evaluate_thresholds(v_records,v_preds,selected['thresholds'])
        if (measured['tp']!=selected['tp'] or measured['fp']!=selected['fp']):
            raise AssertionError('Optimized counts differ from exact classwise evaluation')
    print('Class-specific VALIDATION selection: '+(json.dumps(selected) if selected
          else 'NONE meets target with recall and prediction-count safeguards'),flush=True)
    print('Best attainable with guards: '+json.dumps(selection['best_guarded']),flush=True)
    result=dict(
        status='EXPLORATORY_CLASSWISE_THRESHOLD_TUNING_NOT_INDEPENDENT',
        target_precision=target,min_recall=min_recall,min_predictions=min_predictions,
        tuning_set='186-photo previously exposed AI-annotated validation',
        global_fixed_threshold_reference=reference,
        validation_manifest_sha256=s.sha(val_manifest),validation_receipt_sha256=v_hash,
        model_config_sha256=s.sha(s.CONFIG),original_checkpoints_unchanged=True,
        selection=selection,independent_test=False,
        historical_test=None,
        limitation='Selected thresholds are validation-tuned, may overfit, and do not improve mAP, recall, or school-bin sorting accuracy automatically.')
    if selected and not skip_test:
        print('Scoring preselected vector on historically exposed 248-photo cohort (no retuning)...',flush=True)
        t_records,t_preds,t_hash=p.load_cohort(test_manifest,test_receipt,device,config)
        test_reference=evaluate_thresholds(t_records,t_preds,dict.fromkeys(CLASS_NAMES,.5))
        measured=evaluate_thresholds(t_records,t_preds,selected['thresholds'])
        result['historical_test']=dict(images=len(t_records),
            manifest_sha256=s.sha(test_manifest),receipt_sha256=t_hash,
            baseline_at_conf_050=test_reference,preselected_thresholds=selected['thresholds'],
            fixed_threshold_metrics=measured,
            precision_at_least_target=(measured['precision'] is not None and
                                       measured['precision']>=target),
            caveat='Exposed historical development cohort; NOT independent confirmation.')
    research_audit.verify_freeze(freeze)
    s.save(out,result)
    print(json.dumps(dict(reference_validation=reference,
        validation_selected=selected,best_guarded=selection['best_guarded'],
        historical_test=result['historical_test'],output=str(out)),indent=2))
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation',type=Path,default=EVALUATION)
    parser.add_argument('--historical-test',type=Path,default=p.TEST)
    parser.add_argument('--val-receipt',type=Path)
    parser.add_argument('--test-receipt',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['mps','cpu'],default='mps')
    parser.add_argument('--target',type=float,default=.97)
    parser.add_argument('--min-recall',type=float,default=.60)
    parser.add_argument('--min-predictions',type=int,default=100)
    parser.add_argument('--skip-test',action='store_true')
    args=parser.parse_args()
    receipt=args.test_receipt or (p.KNOWN_TEST_RECEIPT if p.KNOWN_TEST_RECEIPT.exists() else None)
    run(args.validation,args.historical_test,args.val_receipt,receipt,args.output,
        args.device,args.target,args.min_recall,args.min_predictions,args.skip_test)

if __name__=='__main__':
    main()
