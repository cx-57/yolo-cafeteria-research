"""Admit complete, hash-bound AI-reviewed training photos without changing source boxes."""
from pathlib import Path
import hashlib
import math
from PIL import Image,ImageOps

def admit(reviews,downloads,screen,screen_sha,forbidden_shas=(),forbidden_pixels=()):
    sources={r['id']:r for r in downloads}
    checks={r['id']:r for r in screen['records']}
    assert len(sources)==len(downloads) and len(checks)==len(screen['records'])
    assert len({r['id'] for r in reviews})==len(reviews)
    assert {r['id'] for r in reviews}=={r['id'] for r in downloads if r['status']=='unreviewed'}
    assert screen['validation_images']==246
    seen_shas=set(forbidden_shas);seen_pixels=set(forbidden_pixels);added=[]
    for review in reviews:
        if not review['accepted']:continue
        source=sources[review['id']];check=checks[review['id']]
        assert source['status']=='unreviewed' and review['human_verified'] is False
        assert review['duplicate_screen_sha']==screen_sha
        assert not check['hold'] and not check['exact_duplicate']
        path=Path(source['image']);raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        assert digest==source['sha']==review['image_sha']==check['sha']
        assert digest not in seen_shas
        original=source['original'];w,h=original['width'],original['height']
        assert 1162 not in original.get('not_exhaustive_category_ids',[])
        with Image.open(path) as opened:
            assert opened.format=='JPEG' and opened.getexif().get(274,1)==1
            picture=ImageOps.exif_transpose(opened).convert('RGB');assert picture.size==(w,h)
            pixel=hashlib.sha256(str(picture.size).encode()+picture.tobytes()).hexdigest()
        assert raw[-2:]==b'\xff\xd9' and pixel==check['pixels'] and pixel not in seen_pixels
        annotations=source['annotations'];assert annotations and annotations==review['source_annotations']
        indices=review['annotation_indices'];assert sorted(indices)==list(range(len(annotations)))
        rows=[]
        for index in indices:
            annotation=annotations[index];x,y,bw,bh=annotation['bbox']
            assert annotation['category_id']==1162 and annotation['image_id']==original['id']
            assert all(math.isfinite(value) for value in (x,y,bw,bh))
            assert 0<=x<x+bw<=w+.01 and 0<=y<y+bh<=h+.01
            rows.append(f'0 {(x+bw/2)/w:.9f} {(y+bh/2)/h:.9f} {bw/w:.9f} {bh/h:.9f}')
        added.append(dict(id=source['id'],source_image=str(path),image_sha=digest,pixels=pixel,rows=rows,original=original,annotations=annotations,human_verified=False))
        seen_shas.add(digest);seen_pixels.add(pixel)
    return added
