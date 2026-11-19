"""Run frozen specialists on an UNVERIFIED image-review ZIP for exploratory inspection.

No object detection accuracy, sorting accuracy, or independent validation is measured.
Do not train or choose checkpoints on this output; keep independent new-school holdout sealed.
"""
import argparse
import csv
import hashlib
import html
import json
import math
import shutil
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import specialists as s
import disposal as d

MAX_ARCHIVE_UNCOMPRESSED = 650 * 1024 * 1024
MAX_IMAGES = 1500
PALETTE = {
    'compost': '#1f7a4e', 'recycling': '#2463b4',
    'landfill': '#7b4f35', 'review': '#bc7815'
}


def check_frozen(root):
    freeze = json.loads((root / 'configs/research-freeze.json').read_text())
    protected = freeze['protected_sha256']
    targets = ['configs/specialists.json', 'configs/disposal.json',
               *(f'weights/{p}.pt' for p in ('food-container-carton', 'plate-wrapper', 'utensil', 'bottle'))]
    for name in targets:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f'Frozen asset missing: {path} (weights stay on the Mac)')
        if s.sha(path) != protected[name]:
            raise ValueError(f'Frozen asset hash mismatch: {name}. Refusing to run altered experiment.')
    return {name: protected[name] for name in targets}


def unpack_checked(archive, extraction):
    """Extract only declared image files and JSON manifest with byte-level verification."""
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        if z.testzip() is not None:
            raise ValueError('Corrupt ZIP members')
        if sum(m.file_size for m in members) > MAX_ARCHIVE_UNCOMPRESSED:
            raise ValueError('ZIP expands beyond limit')
        names = [m.filename for m in members if not m.is_dir()]
        if len(names) != len(set(names)):
            raise ValueError('ZIP contains repeated names')
        allowed = {'manifest.json', 'index.html', 'README.txt', 'README.md'}
        for m in members:
            name = PurePosixPath(m.filename)
            if name.is_absolute() or '..' in name.parts or '\\' in m.filename:
                raise ValueError('Unsafe ZIP member path')
            if m.is_dir():
                continue
            if m.filename not in allowed and not (name.parts[0] == 'images' and len(name.parts) == 2):
                raise ValueError('Unexpected ZIP member: ' + m.filename)
        if 'manifest.json' not in names:
            raise ValueError('Review ZIP lacks manifest.json')
        manifest = json.loads(z.read('manifest.json'))
        records = manifest.get('images')
        if not isinstance(records, list) or not (1 <= len(records) <= MAX_IMAGES):
            raise ValueError('Invalid manifest image count')
        if manifest.get('independent_certified_images') not in (0, None):
            raise ValueError('Unexpected independent certification; this is an exploratory runner')
        if any(not isinstance(r, dict) for r in records):
            raise ValueError('Invalid image records')
        paths = set()
        hashes = set()
        for row in records:
            path = row.get('path')
            digest = row.get('sha256')
            if (not isinstance(path, str) or path not in names or
                PurePosixPath(path).parts[:1] != ('images',) or
                len(PurePosixPath(path).parts) != 2 or
                not isinstance(digest, str) or len(digest) != 64):
                raise ValueError('Manifest has missing/invalid image identity')
            if path in paths or digest in hashes:
                raise ValueError('Repeated image path or SHA')
            paths.add(path); hashes.add(digest)
            data = z.read(path)
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError('Original image hash mismatch: ' + path)
            target = extraction / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        if {x for x in names if x.startswith('images/')} != paths:
            raise ValueError('Image directory and manifest membership differ')
        (extraction/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def overlay(image, rows, destination):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    with Image.open(image) as img:
        original = ImageOps.exif_transpose(img).convert('RGB')
    w,h=original.size
    d = ImageDraw.Draw(original)
    for result in rows:
        x1,y1,x2,y2 = result['box_xyxy_normalized']
        rect=tuple(round(v * dim) for v,dim in zip((x1,y1,x2,y2),(w,h,w,h)))
        label=f"{result['object_class']} {result['detection_confidence']:.2f} -> {result['category']}"
        color=PALETTE.get(result['category'], '#444444')
        d.rectangle(rect, outline=color, width=max(2,round(min(w,h)/220)))
        d.text((max(0,rect[0]), max(0,rect[1]-18)), label, fill=color, stroke_width=1, stroke_fill='white')
    destination.parent.mkdir(parents=True, exist_ok=True)
    original.save(destination, quality=88)


def summarize(decisions, manifest, output, input_bundle):
    image_by_name = {str((output / 'source' / r['path']).resolve()): r for r in manifest['images']}
    per_class, per_category = Counter(), Counter()
    records = []
    per_image = []
    for key in sorted(decisions):
        if key not in image_by_name:
            raise ValueError('Unexpected inference image identity')
        row = image_by_name[key]
        preds = decisions[key]
        source_name = Path(row['path']).name
        per_image.append({'image':source_name,'source':row.get('source',''),
                          'source_hint':row.get('source_hint',''),
                          'detections':len(preds),'classes':sorted({p['object_class'] for p in preds})})
        for pred in preds:
            per_class[pred['object_class']] += 1
            per_category[pred['category']] += 1
            records.append({
                'image': source_name,'source_hint':row.get('source_hint',''),
                'predicted_class':pred['object_class'],
                'confidence':round(pred['detection_confidence'],6),
                'suggested_category':pred['suggested_category'],
                'actual_output_category':pred['category'],
                'reason_codes':';'.join(pred['reasons']),
                'x1':pred['box_xyxy_normalized'][0],
                'y1':pred['box_xyxy_normalized'][1],
                'x2':pred['box_xyxy_normalized'][2],
                'y2':pred['box_xyxy_normalized'][3],
            })
        overlay(Path(key), preds, output / 'overlays' / (Path(row['path']).stem + '.jpg'))
    columns = ['image','source_hint','predicted_class','confidence','suggested_category',
               'actual_output_category','reason_codes','x1','y1','x2','y2']
    with (output/'detections.csv').open('w',newline='',encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader();writer.writerows(records)
    summary = {
        'status':'EXPLORATORY_INFERENCE_UNVERIFIED_PUBLIC_PHOTOS',
        'images':len(decisions), 'detected_objects_at_display_floor':len(records),
        'images_with_no_displayed_detections':sum(not decisions[n] for n in decisions),
        'detections_by_object_class':dict(sorted(per_class.items())),
        'detections_by_proposed_bin':dict(sorted(per_category.items())),
        'human_verified_object_boxes':0,
        'human_verified_disposal_bin_labels':0,
        'independent_performance_estimated':False,
        'map50':None,'map50_95':None,'actual_correct_bin_accuracy':None,
        'provenance':'UNRESOLVED',
        'warning':('These are counts of MODEL PREDICTIONS, not precision, recall, '
                   'mAP or correct-bin sorting accuracy. Ground truth is unavailable.'),
        'per_image':per_image,
        'source_bundle':str(input_bundle),
    }
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    cards=[]
    for row in per_image:
        stem=Path(row['image']).stem
        pic='overlays/'+stem+'.jpg'
        cards.append('<article><img loading="lazy" src="'+html.escape(pic,quote=True)+'"><p><b>'+
                     html.escape(row['image'])+'</b><br>hint: '+html.escape(row['source_hint'])+
                     '<br>detections: '+str(row['detections'])+'</p></article>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Exploratory detector review</title>
    <style>body{font:16px system-ui,sans-serif;max-width:1200px;margin:25px auto;padding:0 12px;background:#fafafa;color:#222}
    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:15px}
    article{background:white;border:1px solid #ddd;padding:9px;border-radius:6px}
    img{width:100%;height:225px;object-fit:contain}p{overflow-wrap:anywhere}</style>
    <h1>Frozen YOLO: exploratory visual review</h1><p><strong>NOT an independent evaluation.</strong> Predictions only;
    no human verified labels or bin accuracy. Review this page alongside summary.json and detections.csv.</p>
    <div class="grid">'''+''.join(cards)+'''</div></html>'''
    (output/'index.html').write_text(page)
    return summary


def run(bundle, output, device):
    if output.exists():
        raise FileExistsError('Refusing to overwrite existing output: '+str(output))
    if not bundle.is_file():
        raise FileNotFoundError('Review bundle not found: '+str(bundle))
    frozen=check_frozen(s.ROOT)
    output.mkdir(parents=True)
    source=output/'source'
    source.mkdir()
    manifest=unpack_checked(bundle,source)
    records=[{'name':str((source/r['path']).resolve()),
              'image_path':str((source/r['path']).resolve()),
              'sha256':r['sha256']} for r in manifest['images']]
    receipt = s.predict(records,device)
    receipt = s.validate_receipt(receipt,records,s.read(s.CONFIG),s.sha(s.CONFIG))
    policy=s.read(d.DEFAULT_RULES)
    decisions=d.classify_predictions(receipt['predictions'],policy,.25)
    s.save(output/'predictions.json',{
        'policy':policy,'policy_sha256':s.sha(d.DEFAULT_RULES),
        'decisions':decisions,'prediction_receipt':receipt,
        'limitation':'EXPLORATORY ONLY. No independent object or disposal ground truth.'
    })
    if check_frozen(s.ROOT)!=frozen:
        raise ValueError('Frozen model or policy changed during inference')
    return summarize(decisions,manifest,output,bundle)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle',type=Path,help='44-image review bundle ZIP (no model weights)')
    parser.add_argument('--out',type=Path,required=True,help='New output directory')
    parser.add_argument('--device',choices=['cpu','mps'],default='mps')
    args=parser.parse_args()
    result=run(args.bundle.expanduser().resolve(),args.out.expanduser().resolve(),args.device)
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_image',)},indent=2))


if __name__=='__main__':
    main()
