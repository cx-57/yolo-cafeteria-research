"""Assemble the same 44 screened public photos for GitHub Actions artifact download.

Source files come from two previous, immutable GitHub workflow artifacts.
Produces a single-level ZIP *via* upload-artifact, so it is directly consumable
by ./scripts/run_specialists.sh explore <downloaded-artifact.zip>.
"""
import argparse
import csv
import hashlib
import html
import json
import shutil
from pathlib import Path
import rebuild_and_explore_44 as source

def build(trashbox, commons, output):
    if output.exists():
        raise FileExistsError('Refusing to overwrite existing output: '+str(output))
    chosen = []
    for group, folder in (('commons', commons), ('trashbox', trashbox)):
        selected = source.select(folder,group)
        for digest,(record,blob) in sorted(selected.items()):
            name='images/'+group+'__'+record['filename']
            chosen.append((name,digest,record,blob,group))
    if len(chosen)!=44 or len(set(c[1] for c in chosen))!=44:
        raise ValueError('Expected 44 distinct original image bytes')
    output.mkdir(parents=True)
    (output/'images').mkdir()
    manifest=[]
    for name,digest,record,blob,group in chosen:
        path=output/name
        path.write_bytes(blob)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Copied image hash mismatch')
        manifest.append({
            'path':name,'sha256':digest,'source':group,
            'source_hint': record.get('project_class_hint') if group=='commons' else record.get('source_hint'),
            'source_original':record['filename'],
            'source_url':record.get('commons_page','') if group=='commons' else 'https://github.com/nikhilvenkatkumsetty/TrashBox',
            'license':record.get('license','NOT_VERIFIED'),
            'creator':record.get('creator_html',''),
            'license_url':record.get('license_url',''),
            'independent':False,'human_reviewed':False,
            'scoring_eligible':not (group=='trashbox' and record['filename'] in {
                'bottle_09_c8523bf883.jpg','bottle_10_5624478ce4.jpg',
                'bottle_13_f9e2507b47.jpg'}),
            'draft_boxes':[]
        })
    content={
        'schema_version':2,
        'status':'EXPLORATORY_44_PUBLIC_SOURCE_SCREENED_NOT_INDEPENDENT',
        'independent_certified_images':0,
        'human_verified_images':0,
        'verified_bin_labels':0,
        'images':manifest,
    }
    (output/'manifest.json').write_text(json.dumps(content,indent=2)+'\n')
    cards=[]
    for im in manifest:
        cards.append('<article><img loading="lazy" src="'+html.escape(im['path'],quote=True)+'"><p>'+
                     html.escape(Path(im['path']).name)+'<br>'+
                     html.escape(im['source_hint'] or '')+
                     ' — SOURCE HINT ONLY</p></article>')
    page='''<!doctype html><meta charset="utf-8"><title>44 photos - exploratory source review</title>
    <style>body{font:16px system-ui;max-width:1240px;margin:auto;padding:20px}
    main{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}
    article{border:1px solid #ddd;padding:8px}img{width:100%;height:190px;object-fit:contain}
    p{overflow-wrap:anywhere}</style>
    <h2>44 screened public source images</h2><p>Not independent or human-ground-truth labels.
    This is a visual contact gallery; use the frozen-model exploratory command for detector predictions.
    Source attribution and license references appear in manifest.json.</p>
    <main>'''+''.join(cards)+'</main>'
    (output/'index.html').write_text(page)
    (output/'README.txt').write_text(
        '44 source images; identical original image bytes to the previous 44-image review selection.\n'
        'This GitHub artifact ZIP can be passed directly to run_specialists.sh explore.\n'
        'The previously supplied AI-draft annotation editor is not bundled here. This is the input-image ZIP.\n'
        'ZERO independently certified photos or human disposal-bin ground truth. Do not claim mAP or correct-bin accuracy.\n'
        'Photograph rights vary by source; these images are for private screening only.\n'
        'Unzip to view index.html; do NOT unzip to run the explore command.\n')
    print(json.dumps({'images':len(manifest),'source_counts':{x:sum(r['source']==x for r in manifest) for x in ('commons','trashbox')},'sha_verified':True,
                      'status':'PROVISIONAL_NOT_INDEPENDENT','output':str(output)},indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--trashbox',required=True,type=Path)
    a.add_argument('--commons',required=True,type=Path)
    a.add_argument('--out',required=True,type=Path)
    x=a.parse_args()
    build(x.trashbox,x.commons,x.out)
