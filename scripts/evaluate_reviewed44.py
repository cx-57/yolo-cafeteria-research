"""Compare original and V3 seven-class mAP on USER-REVIEWED 44 photos.

The source photos were already inspected during model development; any AP
is DEVELOPMENT DIAGNOSTIC, never independent test generalization.
AI-drafted or partially reviewed labels are REFUSED.
"""
import argparse
import hashlib
import json
import math
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import explore_candidates as frozen
import specialists as s
from storage import ROOT

FORMAT='CAFETERIA_PROVISIONAL_HUMAN_REVIEW_V1'
CLASSES=s.NAMES

def dataset_from_review(bundle, review, folder, include_excluded=False):
    if review.get('format')!=FORMAT or review.get('not_independent') is not True:
        raise ValueError('Expected exported human-review JSON with not_independent=true')
    with zipfile.ZipFile(bundle) as z:
        original=json.loads(z.read('manifest.json'))
        entries=original.get('images',[])
        reviews=review.get('images',[])
        if (len(entries)!=44 or len(reviews)!=44 or
                original.get('classes')!=CLASSES):
            raise ValueError('Expected same 44 photos and seven class names')
        # Two historically distributed ZIPs contain these same 44 photos in
        # different orders. Exact image SHA256 is the identity; position is not.
        # Reject duplicate hashes or ANY missing/unrecognized source photo.
        originals={r['sha256']:r for r in entries}
        by_hash={r['sha256']:r for r in reviews}
        if len(originals)!=44 or len(by_hash)!=44 or set(originals)!=set(by_hash):
            missing=sorted(set(originals)-set(by_hash))
            unexpected=sorted(set(by_hash)-set(originals))
            raise ValueError('Review and ZIP contain DIFFERENT photo bytes: '
                             f'missing {len(missing)}, unexpected {len(unexpected)}; '
                             'use a review export from the SAME 44-image collection')
        records=[];per_class=Counter();skipped=[]
        folder.mkdir(parents=True,exist_ok=True)
        for index,image in enumerate(entries,1):
            marked=by_hash[image['sha256']]
            blob=z.read(image['path'])
            digest=hashlib.sha256(blob).hexdigest()
            if digest!=image['sha256']:
                raise ValueError(f'Image bytes changed for photo {index}')
            if not image.get('scoring_eligible',True) and not include_excluded:
                skipped.append(image['id'])
                continue
            if marked.get('reviewed') is not True:
                raise ValueError(f'Photo {index} is not manually marked fully reviewed: {image["id"]}')
            if not isinstance(marked.get('boxes'),list):
                raise ValueError(f'Photo {index} missing reviewed boxes array')
            lines=[]
            for bi,box in enumerate(marked['boxes'],1):
                if box.get('status')!='HUMAN_REVIEWED':
                    raise ValueError(f'Photo {index}, box {bi}: AI draft not individually confirmed')
                cls=box.get('class_name')
                coords=box.get('box_xyxy_normalized')
                if cls not in CLASSES or not isinstance(coords,list) or len(coords)!=4:
                    raise ValueError(f'Photo {index}, box {bi}: invalid class or box')
                if not all(type(v) in (float,int) and math.isfinite(v) for v in coords):
                    raise ValueError('Nonfinite bounding box')
                x1,y1,x2,y2=coords
                if not (0<=x1<x2<=1 and 0<=y1<y2<=1):
                    raise ValueError('Out-of-bounds or zero-area box')
                c=CLASSES.index(cls)
                lines.append(f'{c} {(x1+x2)/2:.9f} {(y1+y2)/2:.9f} {x2-x1:.9f} {y2-y1:.9f}')
                per_class[cls]+=1
            ext=Path(image['path']).suffix.lower()
            name=f'review44-{index:02d}-{digest[:12]}{ext}'
            path=folder/name
            path.write_bytes(blob)
            records.append(dict(name=name,image_path=str(path),sha256=digest,
                rows=lines,known_classes=list(range(7)),
                source_hint=image.get('source_hint',''),
                review_verified_by_user=True))
    if not records:
        raise ValueError('No user-reviewed, scoring-eligible photos')
    if include_excluded and len(records)!=44:
        raise ValueError('Expected all 44 reviewed photos with --include-excluded')
    if set(per_class)!=set(CLASSES):
        raise ValueError('Every one of seven classes needs at least one verified instance; missing: '+
                         ', '.join(sorted(set(CLASSES)-set(per_class))))
    s.validate_annotations(records,list(range(7)))
    s.verify_images(records)
    return records,dict(total_source_photos=44,scored_photos=len(records),
        excluded_photo_ids=skipped,annotation_objects=dict(per_class),
        human_review_check='user-marked statuses checked, not independently audited',
        independent_test=False)

def summarize(result):
    return dict(AP50=result['AP50'],AP50_95=result['AP50_95'],
                images=result['images'],
                per_class=result['per_class'])

def compare(bundle,review_path,v3_config,output,device,include_excluded):
    if output.exists():
        raise FileExistsError('Refusing to overwrite '+str(output))
    bundle=Path(bundle).expanduser().resolve()
    review_path=Path(review_path).expanduser().resolve()
    v3_config=Path(v3_config).expanduser().resolve()
    frozen_fingerprints=frozen.check_frozen(ROOT)
    review=json.loads(review_path.read_text())
    candidate_config=s.read(v3_config)
    original_config=s.read(s.CONFIG)
    old_models={item['id']:item for item in original_config['models']}
    new_models={item['id']:item for item in candidate_config['models']}
    if set(old_models)!=set(new_models) or any(new_models[k]!=old_models[k] for k in old_models if k!='food-container-carton'):
        raise ValueError('V3 config modified specialist(s) other than food-container-carton')
    primary=new_models['food-container-carton']
    if primary.get('classes')!=['food','hinged_container','drink_carton']:
        raise ValueError('V3 class names mismatch')
    if s.sha(ROOT/primary['weights'])!=primary['sha256']:
        raise ValueError('V3 weights differ from candidate config SHA256')
    with tempfile.TemporaryDirectory(prefix='cafeteria-44-reviewed-') as td:
        records,info=dataset_from_review(bundle,review,Path(td),include_excluded)
        results={}
        for label,config in (('original',s.CONFIG),('v3',v3_config)):
            print(f'Evaluating {label}: {len(records)} identical source photos...',flush=True)
            receipt=s.predict(records,device,config_path=config)
            validated=s.validate_receipt(receipt,records,s.read(config),s.sha(config))
            scored=s.score_predictions(records,validated['predictions'],list(range(7)))
            results[label]=summarize(scored)
        if frozen.check_frozen(ROOT)!=frozen_fingerprints:
            raise ValueError('Frozen model files changed while scoring')
    summary=dict(status='USER_REVIEWED_DEVELOPMENT_DIAGNOSTIC_NOT_INDEPENDENT',
        caveat='44 photos were repeatedly inspected for model development; this is NOT an untouched independent test. Reviewed labels have not been third-party audited.',
        image_audit=info,
        source_bundle_sha256=s.sha(bundle),
        user_review_sha256=s.sha(review_path),
        original_config_sha256=s.sha(s.CONFIG),
        v3_config_sha256=s.sha(v3_config),
        original=results['original'],v3=results['v3'],
        delta_AP50=results['v3']['AP50']-results['original']['AP50'],
        delta_AP50_95=results['v3']['AP50_95']-results['original']['AP50_95'],
        score_is_independent=False)
    s.save(output,summary)
    print(json.dumps(dict(
        images_scored=info['scored_photos'],
        human_review_not_independent=True,
        original_AP50=results['original']['AP50'],
        v3_AP50=results['v3']['AP50'],
        delta_AP50=summary['delta_AP50'],
        original_AP50_95=results['original']['AP50_95'],
        v3_AP50_95=results['v3']['AP50_95'],
        delta_AP50_95=summary['delta_AP50_95'],
        output=str(output)),indent=2))
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True,type=Path,help='44-image AI-draft review ZIP, unchanged image bytes')
    p.add_argument('--review',required=True,type=Path,help='JSON exported AFTER all visible boxes are human-confirmed in index.html')
    p.add_argument('--v3-config',required=True,type=Path,help='V3 candidate-specialists.json with pinned weights')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--device',choices=['cpu','mps'],default='mps')
    p.add_argument('--include-excluded',action='store_true',
                   help='Score all 44 including 3 crowded bottle photos ONLY if every visible instance was exhaustively annotated')
    a=p.parse_args()
    compare(a.bundle,a.review,a.v3_config,a.output,a.device,a.include_excluded)

if __name__=='__main__':
    main()
