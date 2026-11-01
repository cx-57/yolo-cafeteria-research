"""Stage NEW public hinged-container training photo candidates via Wikimedia Commons.

Source category != human label. No auto-training, no independent test claims.
Download individual CC/public-domain candidates, exclude older Commons titles,
preserve licensing and hashes, produce browsable review contact sheet.
"""
import argparse
import csv
import hashlib
import html
import io
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps, ImageDraw
from stage_commons_candidates import category_media, get, is_licensed

SEED = 20261101
GROUPS = [
    ('foam_clamshell_box', 'Expanded polystyrene food boxes', 18),
    ('foam_container_variants', 'Expanded polystyrene food containers', 18),
    ('plastic_food_boxes', 'Plastic food boxes', 10),
    ('disposable_lunch_boxes', 'Disposable lunch boxes', 10),
    ('tray_hard_negatives', 'Expanded polystyrene food trays', 8),
]
# Explicitly diverse known relevant real Commons image titles, with source
# existence verified via public category listings. They still need visual QA.
PRIORITY = [
    'File:Hefty square styrofoam food container closed.JPG',
    'File:Have a Nice Day! styrofoam food container.JPG',
    "File:Dickey's Barbecue Pit Box (23021190555).jpg",
    'File:Food container.jpg',
    'File:Hamaknockers burger.jpg',
    'File:Gyudon for takeout by rhosoi in Cupertino, California.jpg',
    'File:Bison burger from BW\'s.jpg',
    'File:Flounder dinner from Hutton\'s Seafood.jpg',
    'File:Chicken quarter meal, Two Peck Crispy Chicken Cannington, 2026 (01).jpg',
    'File:Food served for the participants 01.jpg',
    'File:Food served for the participants 02.jpg',
    'File:Food served for the participants 03.jpg',
    'File:BreakfastSandwich.jpg',
    'File:Ga Kenkey and fish.jpg',
    'File:Beef Teriyaki from Sarku Japan at the Natick Mall, Natick MA.jpg',
    'File:Beef yakiniku don for takeout by jetalone.jpg',
    'File:Bento of tonkatsu, jiaozi, California Rolls and rice with black sesame.jpg',
    'File:HK Lunch Box Oil deep fried white bread n Minced Beef Spaghetti served with meat sauce Aug-2012.JPG',
]
PRIORITY_INDEX = {name:i for i,name in enumerate(PRIORITY)}

def exclude_prior(path):
    with path.open(newline='',encoding='utf8') as f:
        old=list(csv.DictReader(f))
    titles={r['commons_title'] for r in old}
    shas={r['sha256'] for r in old}
    return titles,shas

def pool_category(category):
    out={}
    for page in category_media(category):
        title=page.get('title','')
        for ii in page.get('imageinfo',[]):
            extra=ii.get('extmetadata',{})
            lic_ok,lic=is_licensed(extra)
            if not lic_ok or not ii.get('mime','').startswith('image/'):continue
            if ii.get('width',0)<450 or ii.get('height',0)<400:continue
            url=ii.get('thumburl') or ii.get('url')
            if not url:continue
            out[title]=(ii,lic)
    return out

def stage(out,prior,per_group=None,seed=SEED):
    if out.exists():raise FileExistsError(out)
    titles,shas=exclude_prior(prior)
    out.mkdir(parents=True)
    (out/'images').mkdir()
    rng=random.Random(seed)
    rows=[]; seen_title=set(titles); seen_sha=set(shas); quotas={}
    for group,category,quota in GROUPS:
        if per_group is not None:quota=min(quota,per_group)
        candidates=pool_category(category)
        ranks=list(candidates)
        rng.shuffle(ranks)
        ranks.sort(key=lambda n:(0, PRIORITY_INDEX[n]) if n in PRIORITY_INDEX else (1,0))
        successes=0; skipped=Counter()
        for title in ranks:
            if successes>=quota:break
            if title in seen_title:
                skipped['old_or_duplicate_title']+=1
                continue
            info,license_short=candidates[title]
            url=info.get('thumburl') or info.get('url')
            try:
                blob=get(url)
                if not 2500<=len(blob)<=9_000_000:
                    skipped['unusable_size']+=1;continue
                with Image.open(io.BytesIO(blob)) as im:
                    im.verify()
                with Image.open(io.BytesIO(blob)) as im:
                    im=ImageOps.exif_transpose(im)
                    if min(im.size)<350:
                        skipped['small_resolution']+=1;continue
                    if im.mode not in ('RGB','L','RGBA','P'):
                        skipped['unsupported_mode']+=1;continue
                    # Normalize to a reproducible RGB JPEG to guard weird formats
                    # and prevent excessively large zip files.
                    im=im.convert('RGB')
                    im.thumbnail((1000,1000),Image.Resampling.LANCZOS)
                    buf=io.BytesIO()
                    im.save(buf,format='JPEG',quality=90,subsampling=0)
                    final=buf.getvalue(); width,height=im.size
                digest=hashlib.sha256(blob).hexdigest()
                if digest in seen_sha:
                    skipped['old_or_duplicate_SHA']+=1;continue
                seen_sha.add(digest);seen_title.add(title)
                name=f'{group}_{successes+1:03d}_{digest[:12]}.jpg'
                (out/'images'/name).write_bytes(final)
                meta=info.get('extmetadata',{})
                rows.append(dict(filename=name,source_hint=group,
                    source_category=category,
                    commons_title=title,
                    commons_page=info.get('descriptionurl',''),
                    original_url=info.get('url',''),
                    download_url=url,
                    license=license_short,
                    license_url=meta.get('LicenseUrl',{}).get('value',''),
                    creator_html=meta.get('Artist',{}).get('value',''),
                    original_download_sha256=digest,
                    packaged_sha256=hashlib.sha256(final).hexdigest(),
                    width=width,height=height,
                    visible_hinge_verified='NO',
                    bounding_boxes_verified='NO',
                    past_train_overlap='NOT_CHECKED',
                    approved_for_training='NO',
                    data_role='NEW_TRAIN_CANDIDATE_NOT_HOLDOUT',
                    visual_review='PENDING'))
                successes+=1
            except Exception as exc:
                skipped['download_or_decode_error']+=1
                print('SKIP',title,str(exc)[:130],flush=True)
        quotas[group]={'goal':quota,'staged':successes,'source_pages':len(candidates),
                       'skips':dict(skipped)}
        print(group,quotas[group],flush=True)
    if len(rows)<15:
        raise RuntimeError('Too few usable new source photos; no artifact to export')
    with (out/'review.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    (out/'manifest.json').write_text(json.dumps(dict(
        status='NEW_HINGED_TRAIN_CANDIDATES_NOT_YET_LABELED',
        independently_certified=0, boxes_annotated=0, train_approved=0,
        previous_commons_titles_excluded=len(titles),
        source_gallery_counts=quotas, images=rows),indent=2)+'\n')
    cols=6;cw=232;ch=205;top=74
    sheet=Image.new('RGB',(cols*cw,top+((len(rows)+cols-1)//cols)*ch),'white')
    d=ImageDraw.Draw(sheet)
    d.text((8,8),f'NEW hinged-container TRAINING CANDIDATES ({len(rows)}) — UNLABELED',fill='black')
    d.text((8,28),'Excludes all previously downloaded Commons titles. Hinges not yet verified. NOT a test set.',fill='black')
    for i,row in enumerate(rows):
        x=(i%cols)*cw;y=top+(i//cols)*ch
        with Image.open(out/'images'/row['filename']) as im:
            thumb=im.copy();thumb.thumbnail((cw-8,ch-56))
        sheet.paste(thumb,(x+(cw-thumb.width)//2,y))
        d.text((x+2,y+ch-52),row['filename'][:30],fill='black')
        d.text((x+2,y+ch-34),row['source_hint'][:28],fill='black')
        title=row['commons_title'][5:]
        d.text((x+2,y+ch-16),title[:34],fill='black')
    sheet.save(out/'contact-sheet.jpg',quality=91)
    cards=[]
    for row in rows:
        file=html.escape(row['filename'],quote=True)
        title=html.escape(row['commons_title'])
        source=html.escape(row['commons_page'],quote=True)
        cards.append(f'<article><img loading="lazy" src="images/{file}"><p>{file}</p>'
                     f'<p>{html.escape(row["source_hint"])}</p>'
                     f'<a href="{source}">{title}</a></article>')
    page='''<!doctype html><meta charset="utf-8"><title>New hinged-container photos</title>
    <style>body{font:15px system-ui;margin:auto;max-width:1400px;padding:18px}
    main{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
    article{border:1px solid #ccc;padding:8px;min-width:0}
    article img{width:100%;height:200px;object-fit:contain}
    p,a{overflow-wrap:anywhere}</style>
    <h2>Unlabeled candidates for improved hinged-container training</h2>
    <p>These images are not verified to contain a hinge. Some are intentional
    tray/plate hard-negative candidates. Before training: review each image,
    annotate every relevant food/clamshell/carton target, verify licenses,
    and scan historical development/validation/test image overlap.</p>
    <main>'''+''.join(cards)+'</main>'
    (out/'index.html').write_text(page)
    (out/'README.md').write_text(
        '# Fresh hinged-container image candidates — NOT LABELED\n\n'
        'All photographs were downloaded via Wikimedia Commons imageinfo and screened '
        'for advertised CC/public-domain licenses. Verify license and attribution per page.\n\n'
        'Prior 38 Commons images excluded by source title, and download byte hashes. '
        'Any older private training images on your Mac remain an unverified overlap risk.\n\n'
        'These are NOT automatically approved for YOLO training. They lack box labels; '
        'some are ordinary trays or food photography, not hinged containers. '
        'Do not use these as test data and then tune on them.\n\n'
        'Open index.html to review. Individual author/license/source in review.csv.\n')
    print(json.dumps(dict(images_staged=len(rows),counts=quotas,
        old_commons_titles_excluded=len(titles),annotated=0,
        approved_for_training=0),indent=2),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--exclude-previous-review',required=True,type=Path)
    p.add_argument('--per-group',type=int)
    a=p.parse_args()
    if a.per_group is not None and a.per_group<1:p.error('per-group must be >=1')
    stage(a.out,a.exclude_previous_review,per_group=a.per_group)

if __name__=='__main__': main()
