"""Admit whole Open Images photos only after scope, geometry, provenance and overlap review."""
from pathlib import Path
import hashlib
import json
import math
from PIL import Image,ImageOps


def admit(reviews,downloads,screen,screen_sha,source_groups,photo_metadata,forbidden_shas=(),forbidden_pixels=()):
    sources={r['id']:r for r in downloads};groups={r['id']:r for r in source_groups}
    metadata={r['ImageID']:r for r in photo_metadata};checks={r['id']:r for r in screen['records']}
    assert len(sources)==len(downloads) and len(groups)==len(source_groups) and len(metadata)==len(photo_metadata)
    assert len(checks)==len(screen['records']) and screen['validation_images']==246
    assert len({r['id'] for r in reviews})==len(reviews)
    assert {r['id'] for r in reviews}=={r['id'] for r in downloads if r['status']=='unreviewed_generic_bottle'}
    seen_shas=set(forbidden_shas);seen_pixels=set(forbidden_pixels);added=[]
    for review in reviews:
        if not review['accepted']:continue
        source=sources[review['id']];group=groups[source['id']];meta=metadata[source['id']];checked=checks[source['id']]
        assert review['full_scene_reviewed'] is True and review['bottle_scope_reviewed'] is True
        assert review['independent_second_review'] is True and review['human_verified'] is False
        assert source['official_split']==meta['Subset']=='train' and meta['Rotation'] in ('0','0.0')
        assert meta['License'].startswith('https://creativecommons.org/') and meta['Author'] and meta['OriginalLandingURL']
        assert source['photo_metadata']==meta and review['photo_metadata_sha']==hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
        assert source['annotations']==group['annotations']==review['source_annotations']
        assert source['positive_source_labels']==group['positive_source_labels']
        assert any(r['ImageID']==source['id'] and r['LabelName'] in {'/m/02rlncx','/m/0118n_9r'} and float(r['Confidence'])==1 for r in source['positive_source_labels'])
        assert review['duplicate_screen_sha']==screen_sha and not checked['hold'] and not checked['exact_duplicate']
        raw=Path(source['image']).read_bytes();digest=hashlib.sha256(raw).hexdigest()
        assert digest==source['sha']==review['image_sha']==checked['sha'] and digest not in seen_shas
        with Image.open(source['image']) as opened:
            assert opened.format=='JPEG' and opened.getexif().get(274,1)==1 and raw[-2:]==b'\xff\xd9'
            picture=ImageOps.exif_transpose(opened).convert('RGB')
            assert picture.size==(source['width'],source['height'])
            pixel=hashlib.sha256(str(picture.size).encode()+picture.tobytes()).hexdigest()
        assert pixel==source['pixels']==checked['pixels'] and pixel not in seen_pixels
        annotations=source['annotations'];indices=review['annotation_indices']
        assert annotations and all(type(i) is int for i in indices) and sorted(indices)==list(range(len(annotations)))
        rows=[]
        for index in indices:
            box=annotations[index]
            assert box['ImageID']==source['id'] and box['LabelName']=='/m/04dr76w' and float(box['Confidence'])==1
            assert box['Source'] in ('xclick','activemil') and box['IsGroupOf']==box['IsDepiction']=='0'
            x,z,y,w=[float(box[key]) for key in ('XMin','XMax','YMin','YMax')]
            assert all(math.isfinite(v) for v in (x,z,y,w)) and 0<=x<z<=1 and 0<=y<w<=1
            rows.append(f'0 {(x+z)/2:.9f} {(y+w)/2:.9f} {z-x:.9f} {w-y:.9f}')
        added.append(dict(id=source['id'],source_image=source['image'],image_sha=digest,pixels=pixel,rows=rows,annotations=annotations,photo_metadata=meta,positive_source_labels=source['positive_source_labels'],human_verified=False))
        seen_shas.add(digest);seen_pixels.add(pixel)
    return added
