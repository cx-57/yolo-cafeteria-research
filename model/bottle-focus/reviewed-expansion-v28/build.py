"""Extend V20 with a separate reviewed round-four training copy; validation stays frozen."""
from pathlib import Path
import importlib.util
import shutil
import yaml
from PIL import Image

RUN=Path(__file__).resolve().parent
BASE=RUN.parent/'context-balance-v20'
REVIEW=RUN.parent/'external-data-v7'
CACHE=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-reviewed-v28')

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

base=load_module('v28_base',BASE/'build.py');m=base.m;s=base.s
admission=load_module('v28_admission',RUN/'admission.py')
PROVENANCE=['round4-visual-reviews.json','lvis-round4-downloads.json','round4-duplicate-screen.json']

def candidates():
    reviews,downloads,screen=[s.load(REVIEW/name) for name in PROVENANCE]
    screen_sha=s.sha(REVIEW/PROVENANCE[2])
    assert screen['validation_images']==246
    assert all(r['duplicate_screen_sha']==screen_sha for r in reviews)
    return admission.admit(reviews,downloads,screen)

def verify():
    base.verify();added=candidates();audit=s.load(RUN/'dataset.json')
    assert audit['base_dataset_audit_sha']==s.sha(BASE/'dataset.json')
    assert audit['provenance_shas']=={name:s.sha(REVIEW/name) for name in PROVENANCE}
    listing=(BASE/'training-exposures.txt').read_text().splitlines()
    for source,record in zip(added,audit['added'],strict=True):
        assert all(record[key]==value for key,value in source.items())
        assert s.sha(Path(record['image']))==record['image_sha']
        assert Path(record['label']).read_text().splitlines()==record['rows']
        listing.extend([record['image']]*4)
    assert len(listing)==1964 and len(set(listing))==1115
    assert listing==(RUN/'training-exposures.txt').read_text().splitlines()
    assert s.sha(RUN/'training-exposures.txt')==audit['training_listing_sha']
    assert yaml.safe_load((RUN/'data.yaml').read_text())==audit['config']
    assert audit['training_unique_bottle_boxes']==471
    assert audit['added_images']==len(audit['added'])==17
    assert audit['added_boxes']==sum(len(r['rows']) for r in audit['added'])==20
    assert audit['total_external_images']==audit['prior_external_images']+audit['added_images']==74
    assert audit['total_external_boxes']==audit['prior_external_boxes']+audit['added_boxes']==129
    assert audit['validation_images']==246 and audit['validation_bottle_boxes']==67
    return audit

if __name__=='__main__':
    source=base.verify();added=candidates();listing=(BASE/'training-exposures.txt').read_text().splitlines()
    existing_names={Path(p).name for p in listing}
    validation_hashes={r['sha'] for r in m.MANIFEST['records'] if r['split']=='val'}
    for r in added:
        original=Path(r['source_image'])
        assert original.name not in existing_names and r['image_sha'] not in validation_hashes
        with Image.open(original) as im:
            assert im.size==(r['original']['width'],r['original']['height']) and im.getexif().get(274,1)==1
            assert im.format=='JPEG' and original.read_bytes()[-2:]==b'\xff\xd9'
            im.verify()
        image=CACHE/'images/train'/original.name;label=CACHE/'labels/train'/f'{original.stem}.txt'
        image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
        if image.exists():assert s.sha(image)==r['image_sha']
        else:shutil.copy2(original,image)
        text='\n'.join(r['rows'])+'\n'
        if label.exists():assert label.read_text()==text
        else:label.write_text(text)
        r.update(image=str(image),label=str(label));listing.extend([str(image)]*4)
    p=RUN/'training-exposures.txt';p.write_text('\n'.join(listing)+'\n')
    config=dict(path=str(m.DATA),train=str(p),val=str(m.DATA/'images/val'),names=['bottle'])
    (RUN/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    audit=dict(source,base_dataset_audit_sha=s.sha(BASE/'dataset.json'),provenance_shas={name:s.sha(REVIEW/name) for name in PROVENANCE},added=added,added_images=17,added_boxes=20,prior_external_images=57,prior_external_boxes=109,total_external_images=74,total_external_boxes=129,round4_added_images=17,round4_added_boxes=20,round4_positive_repeats=4,training_unique_images=1115,training_unique_bottle_boxes=471,training_exposures=1964,training_listing_sha=s.sha(p),config=config,review='AI visual reviews, not human verified; source overlap screening is heuristic',independent_test=False)
    s.save(RUN/'dataset.json',audit);verify();print('V28_DATA_VERIFIED',1115,471,1964,flush=True)
