"""Add reviewed, hash-bound external training photos; keep original validation fixed."""
from pathlib import Path
import hashlib,importlib.util,json,shutil
import yaml
from PIL import Image
RUN=Path(__file__).resolve().parent
CACHE=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-expanded-v13')
spec=importlib.util.spec_from_file_location('bottle_common',RUN.parent/'plastic-specialist-s-v2/run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=m.s

def verify():
    s.validate_configuration(m.DATA,m.MANIFEST['records']);s.validate(m.MANIFEST['records'],m.DATA)
    audit=s.load(RUN/'dataset.json')
    review=RUN.parent/'hard-negative-polish-v11/negative-reviews.json'
    assert s.sha(review)==audit['quality_hold_review_sha']
    held={r['name'] for r in s.load(review)['holds']}
    listing=(RUN/'training-exposures.txt').read_text().splitlines()
    assert not held & {Path(p).name for p in listing}
    assert len(set(listing))==1108 and len(listing)==1651
    assert audit['added_images']==57 and audit['added_boxes']==109
    assert s.sha(RUN/'visual-reviews.json')==audit['visual_review_sha']
    for r in audit['added']:
        assert s.sha(Path(r['image']))==r['image_sha']
        assert Path(r['label']).read_text().splitlines()==r['rows']
    assert s.sha(RUN/'training-exposures.txt')==audit['training_listing_sha']
    assert yaml.safe_load((RUN/'data.yaml').read_text())==audit['config']
    return audit

if __name__=='__main__':
    downloads={r['id']:r for r in s.load(RUN/'lvis-downloads.json') if r['status']=='unreviewed'}
    screen={r['id']:r for r in s.load(RUN/'combined-duplicate-screen.json')['records']}
    added=[]
    for review in s.load(RUN/'visual-reviews.json'):
        if not review['accepted']:continue
        assert not screen[review['id']]['hold']
        r=downloads[review['id']];source=Path(r['image']);assert s.sha(source)==review['image_sha']
        with Image.open(source) as im:
            image_format=im.format
            im.verify()
        with Image.open(source) as im:
            w,h=im.size;assert (w,h)==(r['original']['width'],r['original']['height'])
        if image_format=='JPEG':
            assert source.read_bytes()[-2:]==b'\xff\xd9', 'JPEG normalization required'
        else:
            assert image_format=='PNG', 'Unsupported source format'
        rows=[]
        for i in review['annotation_indices']:
            x,y,bw,bh=r['annotations'][i]['bbox']
            assert 0<=x<x+bw<=w+.01 and 0<=y<y+bh<=h+.01
            rows.append(f'0 {(x+bw/2)/w:.9f} {(y+bh/2)/h:.9f} {bw/w:.9f} {bh/h:.9f}')
        image=CACHE/'images/train'/source.name;label=CACHE/'labels/train'/(source.stem+'.txt')
        image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,image);label.write_text('\n'.join(rows)+'\n')
        added.append(dict(id=r['id'],image=str(image),label=str(label),image_sha=s.sha(image),rows=rows,source=r['original'],group=r['id'],human_verified=False))
    exposures=(RUN.parent/'context-resampling-v5/training-exposures.txt').read_text().splitlines()
    assert len(exposures)==1428 and len(set(exposures))==1056
    holds=s.load(RUN.parent/'hard-negative-polish-v11/negative-reviews.json')['holds']
    originals={r['name']:r for r in m.MANIFEST['records']}
    for r in holds:
        assert originals[r['name']]['split']=='train' and not originals[r['name']]['rows'] and originals[r['name']]['sha']==r['image_sha']
    exposures=[p for p in exposures if Path(p).name not in {r['name'] for r in holds}]
    assert len(exposures)==1423
    for r in added:exposures.extend([r['image']]*4)
    assert len(added)==57 and sum(len(r['rows']) for r in added)==109
    assert len(set(exposures))==1108 and len(exposures)==1651
    listing=RUN/'training-exposures.txt';listing.write_text('\n'.join(exposures)+'\n')
    config=dict(path=str(m.DATA),train=str(listing),val=str(m.DATA/'images/val'),names=['bottle'])
    (RUN/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    s.save(RUN/'dataset.json',dict(quality_hold_review_sha=s.sha(RUN.parent/'hard-negative-polish-v11/negative-reviews.json'),training_quality_holds=holds,added=added,added_images=len(added),added_boxes=sum(len(r['rows']) for r in added),training_unique_images=len(set(exposures)),training_exposures=len(exposures),training_listing_sha=s.sha(listing),config=config,visual_review_sha=s.sha(RUN/'visual-reviews.json'),fixed_source_fingerprint=m.MANIFEST['source_fingerprint'],validation_protocol=m.MANIFEST['validation_protocol'],validation_images=246,validation_bottle_boxes=67,independent_test=False))
    audit=verify();print('BOTTLE_EXTERNAL_DATA_VERIFIED',audit['added_images'],audit['added_boxes'],audit['training_unique_images'],audit['training_exposures'])
