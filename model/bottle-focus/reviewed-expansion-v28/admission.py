"""Admit only reviewed source photos with matching images, annotations and overlap screen."""
from pathlib import Path
import hashlib
import math

def admit(reviews, downloads, screen):
    sources={r['id']:r for r in downloads}
    screened={r['id']:r for r in screen['records']}
    assert len({r['id'] for r in reviews})==len(reviews)
    added=[]
    for review in reviews:
        if not review['accepted']:continue
        r=sources[review['id']];checked=screened[review['id']]
        digest=hashlib.sha256(Path(r['image']).read_bytes()).hexdigest()
        assert digest==r['sha']==review['image_sha']==checked['sha']
        assert not checked['hold'] and not checked['exact_duplicate']
        assert r['annotations']==review['source_annotations']
        indices=review['annotation_indices']
        assert sorted(indices)==list(range(len(r['annotations'])))
        w,h=r['original']['width'],r['original']['height'];rows=[]
        for i in indices:
            ann=r['annotations'][i];x,y,bw,bh=ann['bbox']
            assert ann['category_id']==1162 and ann['image_id']==r['original']['id']
            assert all(math.isfinite(v) for v in (x,y,bw,bh))
            assert 0<=x<x+bw<=w+.01 and 0<=y<y+bh<=h+.01
            rows.append(f'0 {(x+bw/2)/w:.9f} {(y+bh/2)/h:.9f} {bw/w:.9f} {bh/h:.9f}')
        added.append(dict(id=r['id'],source_image=r['image'],image_sha=digest,rows=rows,original=r['original'],annotations=r['annotations'],human_verified=False))
    assert len(added)==17 and sum(len(r['rows']) for r in added)==20
    return added
