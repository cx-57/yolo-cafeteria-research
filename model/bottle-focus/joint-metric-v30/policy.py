"""Exact joint target and nondominated metric pairs; neither metric hides the other."""
import math
import hashlib
import json
from pathlib import Path

def metrics(report):
    a,b=report['AP50'],report['AP50_95']
    if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (a,b)) or not 0<=b<=a<=1:
        raise ValueError('Invalid AP50/AP50-95 pair')
    return a,b

def target_met(report):
    a,b=metrics(report)
    return a>=.90 and b>=.64

def save_roles(current,previous):
    a,b=metrics(current);roles=set()
    if 'ap50' not in previous or a>metrics(previous['ap50'])[0]:roles.add('ap50')
    if 'localization' not in previous or b>metrics(previous['localization'])[1]:roles.add('localization')
    if target_met(current):roles.add('joint')
    return roles

def frontier(reports):
    reports=list(reports)
    result=[];seen=set()
    for r in reports:
        a,b=metrics(r)
        if (a,b) in seen:continue
        if any((x>=a and y>=b and (x>a or y>b)) for x,y in map(metrics,reports)):
            continue
        seen.add((a,b));result.append(r)
    return result

def verified_records(run):
    audit=json.loads((run/'dataset.json').read_text())
    path=run/'metrics-history.json'
    reports=json.loads(path.read_text()) if path.exists() else []
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for r in reports:
        metrics(r)
        assert sha(Path(r['weights']))==r['weights_sha']
        assert r['training_dataset_audit_sha']==sha(run/'dataset.json')
        assert r['source_dataset_fingerprint']==audit['fixed_source_fingerprint']
        assert r['validation_protocol']==audit['validation_protocol']
        assert r['validation_images']==246 and r['bottle_objects']==67
        assert r['imgsz']==640 and r['batch']==4 and r['nms_iou']==.5 and r['test_time_augmentation'] is False
    return reports

def phase_outcome(previous,reports,epoch):
    assert reports
    points=list(map(metrics,reports));peaks=[max(p[0] for p in points),max(p[1] for p in points)]
    reference=previous.get('reference',list(points[0]));stale=previous.get('stale_phases',0)
    if any(p>=r+.002 for p,r in zip(peaks,reference)):
        reference=peaks;stale=0
    else:stale+=1
    status='target_verified' if any(target_met(r) for r in reports) else 'schedule_complete_below_target' if epoch>=40 else 'screen_plateau_below_target' if stale>=3 else 'running'
    return dict(reference=reference,stale_phases=stale,status=status,best_AP50=peaks[0],best_AP50_95=peaks[1],completed_epoch=epoch)
