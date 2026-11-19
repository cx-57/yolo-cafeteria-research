"""Explore 97% MICRO object-detection precision without changing frozen weights.

Select ONE global display-confidence threshold on the development validation set.
Then evaluate the fixed threshold on historically exposed 248-image test.
This is not independent test generalization, disposal correctness, or an AP gain.
"""
import argparse
import json
import math
from pathlib import Path

import research_audit
import specialists as s
from storage import ARTIFACTS, EVALUATION, ROOT

ALL_CLASSES=list(range(7))
TEST=ARTIFACTS/'test-label-audit-20261031/audited-all248.json'
KNOWN_TEST_RECEIPT=ARTIFACTS/'bottle-retention-20261031/test-active-confirmation.json'

def thresholds():
    # Deliberately fixed before seeing test outcomes; .50 historic reference.
    return [round(i/100,2) for i in range(50,100)]+[.995]

def select_threshold(validation_rows,target=.97,min_recall=.60,min_predictions=100):
    if (not 0 < target <= 1 or not 0 <= min_recall <= 1
            or type(min_predictions) is not int or min_predictions < 1):
        raise ValueError('Invalid selection constraints')
    eligible=[r for r in validation_rows if r['precision'] is not None
              and r['precision'] >= target
              and r['recall'] is not None and r['recall'] >= min_recall
              and r['predictions'] >= min_predictions]
    # Maximize recall, then precision, then minimize confidence; NEVER test-select.
    return (max(eligible,key=lambda r:(r['recall'],r['precision'],-r['threshold']))
            if eligible else None)

def summarize(records,predictions,threshold):
    out=s.recognition_at_threshold(records,predictions,confidence=threshold)
    return dict(threshold=threshold,**out['micro'])

def load_cohort(manifest_path,receipt_path,device,model_config):
    data=s.read(manifest_path)
    if data.get('names')!=s.NAMES or set(data.get('evaluated_classes',[]))!=set(ALL_CLASSES):
        raise ValueError('Manifest must cover all seven object classes')
    records=data['records']
    s.validate_annotations(records,ALL_CLASSES)
    s.verify_images(records)
    receipt=s.read(receipt_path) if receipt_path else s.predict(records,device,s.CONFIG)
    verified=s.validate_receipt(receipt,records,model_config,s.sha(s.CONFIG))
    return records,verified['predictions'],(s.sha(receipt_path) if receipt_path else None)

def run(val_manifest,test_manifest,val_receipt,test_receipt,
        out,device,target,min_recall,min_predictions,skip_test):
    if out.exists():
        raise FileExistsError('Refusing to overwrite '+str(out))
    freeze=s.read(ROOT/'configs/research-freeze.json')
    research_audit.verify_freeze(freeze)
    config=s.read(s.CONFIG)
    for spec in config['models']:
        if s.sha(ROOT/spec['weights'])!=spec['sha256']:
            raise ValueError('Frozen checkpoint mismatch: '+spec['id'])

    print('Loading 186-photo VALIDATION predictions (selection only)',flush=True)
    v_records,v_preds,v_receipt_hash=load_cohort(val_manifest,val_receipt,device,config)
    rows=[summarize(v_records,v_preds,t) for t in thresholds()]
    fixed=select_threshold(rows,target,min_recall,min_predictions)
    baseline=next(r for r in rows if r['threshold']==.50)
    print('Validation baseline @0.50: '+json.dumps({key:baseline[key] for key in ('precision','recall','predictions')}),flush=True)
    print('Selected on validation ONLY: '+(json.dumps(fixed) if fixed else 'NONE meets precision, recall AND coverage constraints'),flush=True)

    result=dict(status='DEVELOPMENT_CONFIDENCE_DIAGNOSTIC_NOT_INDEPENDENT',
        goal='Object detection micro precision, NOT mAP50 or correct disposal bin accuracy',
        annotation_status='Existing AI-assisted validation/test labels; historical model exposure',
        model_config_sha256=s.sha(s.CONFIG),
        val_manifest_sha256=s.sha(val_manifest),
        val_receipt_sha256=v_receipt_hash,
        selection_constraints=dict(target_micro_precision=target,
                                   min_micro_recall=min_recall,
                                   min_predictions=min_predictions),
        validation=dict(images=len(v_records),baseline_at_conf_050=baseline,
                        sweep=rows,selected_on_validation=fixed),
        test=None,
        threshold_changed_frozen_weights=False,
        independent_test=False)
    if not skip_test:
        if fixed is None:
            print('No validation threshold qualifies. Test not evaluated to avoid chasing historical test results.',flush=True)
        else:
            print('Checking frozen selected threshold on the exposed historical TEST; no reselection',flush=True)
            t_records,t_preds,t_receipt_hash=load_cohort(test_manifest,test_receipt,device,config)
            test_at_selected=summarize(t_records,t_preds,fixed['threshold'])
            test_baseline=summarize(t_records,t_preds,.50)
            result['test']=dict(images=len(t_records),test_manifest_sha256=s.sha(test_manifest),
                test_receipt_sha256=t_receipt_hash,
                threshold_selected_without_test=fixed['threshold'],
                baseline_at_conf_050=test_baseline,
                fixed_threshold_metrics=test_at_selected,
                test_precision_at_least_target=(test_at_selected['precision'] is not None and
                                               test_at_selected['precision']>=target),
                caveat='Exposed historical test, NOT untouched/independent; threshold tuned solely on validation')
    research_audit.verify_freeze(freeze)
    s.save(out,result)
    show=dict(validation_baseline=baseline,validation_selected=fixed,
        historical_test=result['test'],output=str(out),
        caveat='A higher confidence cutoff trades recall for precision; no checkpoint retrained; no independent result')
    print(json.dumps(show,indent=2))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation',type=Path,default=EVALUATION)
    p.add_argument('--historical-test',type=Path,default=TEST)
    p.add_argument('--val-receipt',type=Path,
                   help='Reuse frozen validation predictions; omit to run native YOLO on 186 photos')
    p.add_argument('--test-receipt',type=Path,
                   help='Reuse frozen 248-image predictions; omit to auto-use known receipt if it exists, else run native YOLO')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['mps','cpu'],default='mps')
    p.add_argument('--target',type=float,default=.97)
    p.add_argument('--min-recall',type=float,default=.60)
    p.add_argument('--min-predictions',type=int,default=100)
    p.add_argument('--skip-test',action='store_true',
                   help='Calculate validation diagnostic only; no historical test inference')
    args=p.parse_args()
    test_receipt=(args.test_receipt if args.test_receipt else
                  KNOWN_TEST_RECEIPT if KNOWN_TEST_RECEIPT.is_file() else None)
    run(args.validation,args.historical_test,args.val_receipt,test_receipt,
        args.output,args.device,args.target,args.min_recall,args.min_predictions,args.skip_test)

if __name__=='__main__':
    main()
