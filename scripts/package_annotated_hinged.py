"""Build a TRAIN-ONLY 3-class YOLO image/label ZIP from fixed 64-photo source pool.

Input rights are as-advertised per Commons metadata. Bounding boxes are model-
assisted VISUAL DRAFTS, not verified ground truth or evaluation data.
"""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw
from io import BytesIO

ROOT=Path(__file__).resolve().parents[1]
ANNOTATIONS=ROOT/'configs/hinged-box-drafts-20261101.json'
COLORS={0:'#21a04d',1:'#ed3535',2:'#3764da'}

def package(source,out):
    source=source.resolve()
    out=out.resolve()
    if out.exists():
        raise FileExistsError(out)
    config=json.loads(ANNOTATIONS.read_text())
    manifest=json.loads((source/'manifest.json').read_text())
    if len(manifest['images'])!=config['source_count']:
        raise ValueError('Original source photo order or count changed')
    if len(config['annotations'])!=29 or config['classes']!=['food','hinged_container','drink_carton']:
        raise ValueError('Expected immutable 29-photo three-class visual annotation')
    rows=[]; totals={str(i):0 for i in range(3)}
    out.mkdir(parents=True)
    (out/'images/train').mkdir(parents=True)
    (out/'labels/train').mkdir(parents=True)
    (out/'previews').mkdir(parents=True)
    for key,boxes in config['annotations'].items():
        index=int(key)
        if not 1<=index<=64:
            raise ValueError('Out-of-range source index')
        src=manifest['images'][index-1]
        filename=src['filename']
        image_path=source/'images'/filename
        blob=image_path.read_bytes()
        if hashlib.sha256(blob).hexdigest()!=src['packaged_sha256']:
            raise ValueError('Source image checksum mismatch: '+filename)
        preview=Image.open(BytesIO(blob)).convert('RGB')
        w,h=preview.size
        draw=ImageDraw.Draw(preview)
        label=[]
        for box in boxes:
            if len(box)!=5:
                raise ValueError('Bad label row')
            cls,x1,y1,x2,y2=box
            if type(cls) is not int or cls not in [0,1,2] or not 0<=x1<x2<=1 or not 0<=y1<y2<=1:
                raise ValueError('Invalid class/normalized coordinates')
            totals[str(cls)]+=1
            xc=(x1+x2)/2;yc=(y1+y2)/2;bw=x2-x1;bh=y2-y1
            label.append(f'{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}')
            x0,y0,xr,yr=[int(v) for v in [w*x1,h*y1,w*x2,h*y2]]
            draw.rectangle((x0,y0,xr,yr),outline=COLORS[cls],width=max(3,min(w,h)//180))
            title=config['classes'][cls]
            draw.rectangle((x0,max(0,y0-21),min(w,x0+len(title)*8),y0+1),fill=COLORS[cls])
            draw.text((x0,max(0,y0-20)),title,fill='white')
        (out/'images/train'/filename).write_bytes(blob)
        label_name=Path(filename).with_suffix('.txt').name
        label_blob=('\n'.join(label)+('\n' if label else '')).encode()
        (out/'labels/train'/label_name).write_bytes(label_blob)
        preview.thumbnail((900,900))
        preview.save(out/'previews'/filename,quality=91)
        rows.append(dict(
          source_gallery_index=index,
          image='images/train/'+filename,
          label='labels/train/'+label_name,
          image_sha256=hashlib.sha256(blob).hexdigest(),
          label_sha256=hashlib.sha256(label_blob).hexdigest(),
          source_title=src['commons_title'],
          source_page=src.get('commons_page',''),
          creator_html=src.get('creator_html',''),
          license=src['license'],
          license_url=src.get('license_url',''),
          source_download_sha256=src['original_download_sha256'],
          status='likely_hinged' if any(b[0]==1 for b in boxes) else 'hard_negative',
          boxes=[{'class_id':b[0],'normalized_xyxy':b[1:]} for b in boxes],
          human_verified=False))
    stats=dict(photos=len(rows),clamshell_photo_candidates=sum(r['status']=='likely_hinged' for r in rows),
        hard_negative_photo_candidates=sum(r['status']=='hard_negative' for r in rows),
        box_totals=totals, human_verified_boxes=0, data_role='TRAIN_ONLY_NOT_FIELD_HOLDOUT',
        source_manifest_count=64, old_source_overlap='UNCHECKED_BEYOND_PREVIOUS_COMMONS_TITLE_EXCLUSION')
    if stats['photos']!=29 or stats['box_totals']!={'0':28,'1':24,'2':0}:
        raise ValueError('Expected 29 images with 28 food and 24 hinged draft boxes')
    complete=dict(schema_version=1,status=config['annotation_status'],
                  classes=config['classes'],stats=stats,images=rows,
                  note='Draft training labels; class and box corrections may be necessary. No verified accuracy.')
    (out/'manifest.json').write_text(json.dumps(complete,indent=2)+'\n')
    (out/'classes.txt').write_text('\n'.join(config['classes'])+'\n')
    (out/'README.txt').write_text(
       'AI-visual YOLO draft labels, NOT human verified.\n'
       'Use ONLY as separate development training data. Do NOT score these as holdout.\n'
       'class 0 food, 1 hinged_container, 2 drink_carton.\n'
       'Source rights and creator per manifest entry, verify before redistribution.\n'
       'Original training/validation/test near-duplicate history unresolved.\n')
    import html
    blocks=[]
    for row in rows:
        name=html.escape(Path(row['image']).name,quote=True)
        source_url=html.escape(row['source_page'],quote=True)
        blocks.append('<article><img src="previews/'+name+'" loading="lazy">'
           +'<p>'+html.escape(row['status'])+'</p><a href="'+source_url+'">'
           +html.escape(row['source_title'])+'</a></article>')
    (out/'index.html').write_text('''<!doctype html><meta charset="utf-8">
<title>Draft hinged annotations</title>
<style>body{font:15px system-ui;max-width:1300px;margin:auto;padding:16px}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}
article{border:1px solid #ccc;padding:8px}
img{width:100%;height:200px;object-fit:contain}p,a{overflow-wrap:anywhere}</style>
<h2>29 AI-visual annotated training photos (not human verified)</h2>
<p>Red: hinged_container, green: food. One photo has a deliberately empty label.
License/source for each photo in manifest.json.</p>
<main>'''+''.join(blocks)+'</main>')
    print(json.dumps(stats,indent=2))
    return stats

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True,
                    help='Folder containing the extracted 64-photo artifact')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    package(a.source,a.out)
