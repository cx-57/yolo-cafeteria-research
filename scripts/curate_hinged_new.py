"""Curate the image-screened new Commons photos; do NOT infer true boxes.

This selection comes from a visual contact-sheet inspection of an immutable
64-photo downloaded GitHub Actions artifact. Image roles are PROVISIONAL:
21 clamshell candidate photos, 8 intentional non-hinged hard negatives.
"""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
from PIL import Image, ImageDraw

CLAMSHELL = [1,2,3,4,5,6,9,11,12,13,15,16,17,18,19,20,22,50,54,55,56]
NEGATIVE = [30,36,37,39,48,58,59,64]
# Numeric indices refer to the immutable 64-photo original artifact manifest
EXPECTED_SOURCE = 64

def package(source, output):
    if output.exists():
        raise FileExistsError(output)
    with zipfile.ZipFile(source) as archive:
        manifest=json.loads(archive.read('manifest.json'))
        imgs=manifest['images']
        if len(imgs)!=EXPECTED_SOURCE:
            raise ValueError('Unexpected upstream photo count; refusing ambiguous index selection')
        if set(CLAMSHELL)&set(NEGATIVE) or len(CLAMSHELL)!=21 or len(NEGATIVE)!=8:
            raise ValueError('Invalid frozen visual selection')
        output.mkdir(parents=True)
        (output/'images').mkdir()
        selected=[]
        for role,ids in [('PROVISIONAL_LIKELY_HINGED_CONTAINER',CLAMSHELL),
                         ('PROVISIONAL_HARD_NEGATIVE',NEGATIVE)]:
            for index in ids:
                r=imgs[index-1]
                name=r['filename']
                data=archive.read('images/'+name)
                if hashlib.sha256(data).hexdigest()!=r['packaged_sha256']:
                    raise ValueError('Source image hash mismatch: '+name)
                destination=output/'images'/name
                destination.write_bytes(data)
                selected.append(dict(r,stage_role=role,
                                     source_gallery_index=index,
                                     boxes_verified=False,
                                     train_approved=False))
    preview=Image.new('RGB',(5*242,((len(selected)+4)//5)*224+80),'white')
    d=ImageDraw.Draw(preview)
    d.text((12,10),'29 curated source photos (21 LIKELY clamshells, 8 NEGATIVES)',fill='black')
    d.text((12,35),'AI-visual provisional screening only. Boxes NOT annotated or source history verified.',fill='black')
    for i,row in enumerate(selected):
        im=Image.open(output/'images'/row['filename']).convert('RGB')
        im.thumbnail((230,170))
        x=i%5*242;y=80+i//5*224
        preview.paste(im,(x+(242-im.width)//2,y))
        caption=f'{row["source_gallery_index"]:02d} {"LIKELY_HINGE" if "HINGED" in row["stage_role"] else "NEGATIVE"}'
        d.text((x+5,y+172),caption,fill='black')
        d.text((x+5,y+191),row['filename'][:32],fill='black')
    preview.save(output/'contact-sheet.jpg',quality=90)
    report=dict(
        status='AI_VISUALLY_SCREENED_PROVISIONAL_TRAINING_CANDIDATES',
        source='Wikimedia Commons, sourced from verified 64-photo GitHub artifact',
        photos=len(selected),clamshell_photo_candidates=len(CLAMSHELL),
        negative_photo_candidates=len(NEGATIVE),
        license_pages_must_be_checked=True,original_train_overlap_unverified=True,
        human_verified_bounding_boxes=0,ready_to_train=False,
        cannot_be_used_as_an_independent_holdout=True,
        images=selected)
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    import html
    cards=[]
    for r in selected:
        n=html.escape(r['filename'],quote=True)
        cards.append('<article><img loading="lazy" src="images/'+n+'"><p>'+
                    html.escape(r['stage_role'])+'</p><a href="'+html.escape(r['commons_page'],quote=True)+'">'+
                    html.escape(r['commons_title'])+'</a></article>')
    (output/'index.html').write_text('''<!doctype html><meta charset="utf-8">
    <title>29 curated hinged container photo candidates</title>
    <style>body{font:15px system-ui;max-width:1400px;margin:auto;padding:16px}
    main{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
    article{border:1px solid #aaa;padding:9px}
    img{width:100%;height:190px;object-fit:contain}p,a{overflow-wrap:anywhere}</style>
    <h2>21 likely clamshells + 8 hard-negative photo candidates</h2>
    <p>AI visual screening only; NOT human box labels; don't train until classes
    and boxes are verified. Hard negatives intentionally resemble containers.</p>
    <main>'''+''.join(cards)+'</main>')
    (output/'README.txt').write_text(
        '29 candidate photos, all distinct from earlier 38 Commons source titles.\n'
        '21 AI-visually selected likely hinged containers and 8 hard negatives.\n'
        'These are not labeled or certified training records. Annotate and verify\n'
        'every relevant object and check against all Mac training/validation/test sources.\n'
        'Check each Commons image license, creator and attribution before reuse.\n'
        'DO NOT use as an independent holdout after tuning models against them.\n')
    print(json.dumps({k:v for k,v in report.items() if k!='images'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    package(a.source,a.out)
