"""Compare original vs hinged candidate on the same 44 public review images.

SOURCE-HINT image-level class hits are only diagnostic. Never report these as
recall, mAP, independent accuracy, or actual disposal-bin correctness.
"""
import argparse
import json
import tempfile
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO

import explore_candidates as frozen
import specialists as s
from storage import ROOT

MAPPING = ['food','hinged_container','drink_carton']


def hint_matches(hint, classes):
    if hint == 'hinged_container_candidate':
        return 'hinged_container' in classes
    if hint in ('drink_carton','bottle','wrapper','utensil','food','plate'):
        return hint in classes
    return False


def run(bundle, candidate, output, device='mps'):
    if output.exists():
        raise FileExistsError('Refusing to overwrite output: '+str(output))
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    immutable=frozen.check_frozen(ROOT)
    candidate_sha=s.sha(candidate)
    output.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as tmp:
        image_dir=Path(tmp)/'source'
        manifest=frozen.unpack_checked(bundle,image_dir)
        models=[
            ('frozen',YOLO(str(ROOT/'weights/food-container-carton.pt'))),
            ('candidate',YOLO(str(candidate)))]
        for _,model in models:
            names=model.names
            if isinstance(names,dict):names=[names[i] for i in range(len(names))]
            if names!=MAPPING:raise ValueError('Checkpoint classes do not match three-class specialist')
        counts={name:{'hinged_container_candidate':0,'drink_carton':0} for name,_ in models}
        total={}
        results=[]
        for index,entry in enumerate(manifest['images']):
            image=image_dir/entry['path']
            hint=entry.get('source_hint','')
            total[hint]=total.get(hint,0)+1
            pic=ImageOps.exif_transpose(Image.open(image)).convert('RGB')
            width,height=pic.size
            panels=[]
            sample={'file':entry['path'],'sha256':entry['sha256'],'source_hint':hint}
            for name,model in models:
                prediction=model.predict(str(image),device=device,imgsz=640,batch=1,
                        augment=False,conf=0.25,iou=0.70,max_det=300,half=False,
                        verbose=False)[0]
                boxes=[]
                panel=pic.copy()
                draw=ImageDraw.Draw(panel)
                for row in prediction.boxes.data.cpu().tolist():
                    class_id=int(row[5]); confidence=float(row[4])
                    if class_id not in (0,1,2):
                        raise ValueError('Unexpected checkpoint output class')
                    cls=MAPPING[class_id]
                    boxes.append({'class':cls,'confidence':round(confidence,4),
                                  'box_xyxy_pixels':[round(v,2) for v in row[:4]]})
                    color='#d12d2d' if cls=='hinged_container' else '#2463b4'
                    draw.rectangle(row[:4],outline=color,width=max(2,min(width,height)//250))
                    draw.text((row[0],max(0,row[1]-16)),f'{cls} {confidence:.2f}',fill=color)
                classes={box['class'] for box in boxes}
                sample[name]={'detections':boxes,'source_hint_class_detected':hint_matches(hint,classes)}
                if hint in counts and sample[name]['source_hint_class_detected']:
                    counts[name][hint]+=1
                panels.append(panel)
            if hint=='hinged_container_candidate':
                # Save compact side-by-side visual comparison for all 12 source scenes.
                wide=Image.new('RGB',(width*2,height+26),'white')
                top=ImageDraw.Draw(wide)
                top.text((4,4),'FROZEN',fill='black')
                top.text((width+4,4),'HINGED CANDIDATE',fill='black')
                wide.paste(panels[0],(0,26));wide.paste(panels[1],(width,26))
                wide.thumbnail((1600,1000))
                target=output / 'hinged-overlays' / f'{index:03}_{Path(entry["path"]).stem}.jpg'
                target.parent.mkdir(parents=True,exist_ok=True)
                wide.save(target,quality=88)
            results.append(sample)
        if s.sha(candidate)!=candidate_sha or frozen.check_frozen(ROOT)!=immutable:
            raise ValueError('Candidate/frozen checkpoint bytes changed while comparing')
    result={
        'status':'EXPLORATORY_SOURCE_HINT_DIAGNOSTIC_NOT_ACCURACY',
        'images':len(results), 'source_hint_image_counts':total,
        'class_hits_by_source_hint':counts,
        'hinged_source_hint_delta':(counts['candidate']['hinged_container_candidate']
                                   -counts['frozen']['hinged_container_candidate']),
        'candidate_checkpoint_sha256':candidate_sha,
        'original_checkpoint_sha256':immutable['weights/food-container-carton.pt'],
        'automatic_promotion':False,'independent_accuracy':None,
        'warning':'Not recall or accuracy; public source hints are not verified object ground truth. The 12 hinged photos have already been inspected, so this is development-only diagnosis.',
        'per_image':results
    }
    (output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','mps'],default='mps')
    a=p.parse_args()
    result=run(a.bundle.expanduser().resolve(),
               a.candidate.expanduser().resolve(),
               a.output.expanduser().resolve(),a.device)
    print(json.dumps({k:v for k,v in result.items() if k!='per_image'},indent=2))


if __name__=='__main__':main()
