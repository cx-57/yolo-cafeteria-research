"""Whole-photo admission for reviewed source data, separate from staging/screening."""
from pathlib import Path
from urllib.parse import urlsplit
import hashlib
import json
import math
from PIL import Image,ImageOps

def json_sha(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def hashes(path):
 raw=Path(path).read_bytes()
 with Image.open(path) as opened:picture=ImageOps.exif_transpose(opened).convert('RGB')
 return hashlib.sha256(raw).hexdigest(),hashlib.sha256(str(picture.size).encode()+picture.tobytes()).hexdigest()

def admit(source,review,official,checked,forbidden_ids=(),forbidden_shas=(),forbidden_pixels=()):
 assert review['accepted'] is True and review['full_scene_reviewed'] is True and review['bottle_scope_reviewed'] is True
 assert review['independent_second_review'] is True and review['human_verified'] is False
 assert source['id'] not in forbidden_ids and source['id']==checked['id'] and not checked['hold']
 digest,pixel=hashes(source['image'])
 assert digest==source['sha']==review['image_sha']==checked['sha'] and digest not in forbidden_shas
 assert pixel==checked['pixels'] and pixel not in forbidden_pixels
 with Image.open(source['image']) as picture:picture.verify()
 with Image.open(source['image']) as picture:
  assert picture.format in ('JPEG','PNG') and picture.getexif().get(274,1)==1
  w,h=picture.size
 assert (w,h)==(source['width'],source['height'])
 annotations=source['annotations'];metadata=source['photo_metadata']
 assert annotations==official['annotations']==review['source_annotations']
 assert metadata==official['photo_metadata'] and json_sha(metadata)==review['metadata_sha']
 if source['source_type']=='openimages':
  assert metadata['ImageID']==source['id'] and metadata['Subset']=='train' and metadata['Rotation'] in ('0','0.0')
  assert metadata['License'].startswith('https://creativecommons.org/licenses/') and metadata['Author'] and metadata['OriginalLandingURL']
 elif source['source_type']=='taco':
  assert source['role'] in ('positive','negative') and source['original']==official['original']
  assert (w,h)==(official['original']['width'],official['original']['height'])
  assert metadata['license_url'].startswith('https://creativecommons.org/licenses/') and metadata['author']['name'] and metadata['photo_page']
  assert metadata['photo_id']==Path(urlsplit(source['original']['flickr_url']).path).name.split('_')[0].split('.')[0]
  assert source['all_source_annotations']==official['all_source_annotations']
  if source['role']=='negative':
   assert not any(a['category_id'] in {4,5,6} for a in official['all_source_annotations']), 'Bottle classes require explicit source-scope handling'
  assert not any(a['category_id']==4 for a in official['all_source_annotations']), 'Other-plastic scope has not been separately reviewed'
  assert annotations==[a for a in official['all_source_annotations'] if a['category_id']==5]
 elif source['source_type']=='commons':
  assert source['role']=='negative' and not annotations and source['id']=='commons_'+str(metadata['pageid'])
  assert len(metadata['imageinfo'])==1
  info=metadata['imageinfo'][0];license_url=info['extmetadata']['LicenseUrl']['value'];artist=info['extmetadata']['Artist']['value']
  assert license_url.startswith(('https://creativecommons.org/','http://creativecommons.org/')) and artist
  assert source['source_image_url']==info.get('thumburl',info['url'])
  assert (w,h)==(info.get('thumbwidth',info['width']),info.get('thumbheight',info['height']))
 else:raise AssertionError('Unknown source type')
 if source['role']=='negative':
  assert review.get('verified_absence') is True and review.get('all_source_boxes_non_target') is True
  assert review['annotation_indices']==[]
  rows=[]
 else:
  assert source['role']=='positive' and annotations
  assert all(type(i) is int for i in review['annotation_indices']) and sorted(review['annotation_indices'])==list(range(len(annotations)))
  rows=[]
  for box in annotations:
   if source['source_type']=='openimages':
    assert box['ImageID']==source['id'] and box['LabelName']=='/m/04dr76w' and float(box['Confidence'])==1
    assert box['IsGroupOf']==box['IsDepiction']=='0'
    x,z,y,t=[float(box[k]) for k in ['XMin','XMax','YMin','YMax']]
   else:
    assert box['image_id']==source['original']['id'] and box['category_id']==5
    xx,yy,bw,bh=box['bbox'];x,z,y,t=xx/w,(xx+bw)/w,yy/h,(yy+bh)/h
   assert all(math.isfinite(v) for v in (x,z,y,t)) and 0<=x<z<=1 and 0<=y<t<=1
   rows.append(f'0 {(x+z)/2:.9f} {(y+t)/2:.9f} {z-x:.9f} {t-y:.9f}')
 return dict(id=source['id'],source_image=source['image'],image_sha=digest,pixels=pixel,rows=rows,
             source_type=source['source_type'],role=source['role'],annotations=annotations,photo_metadata=metadata,human_verified=False)
