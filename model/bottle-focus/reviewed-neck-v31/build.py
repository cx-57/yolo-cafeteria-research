"""Build full-image reviewed training data; the capture benchmark remains frozen."""
from pathlib import Path
from collections import Counter
import importlib.util
import shutil
import yaml
import experiment

RUN=Path(__file__).resolve().parent
BASE=RUN.parent/'joint-metric-v30'
REVIEW=RUN.parent/'external-data-v7'
CACHE=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-reviewed-neck-v31')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
base=module('v31_base',BASE/'build.py');m=base.m;s=base.s
admission=module('v31_admission',RUN/'admission.py')
PROVENANCE=[f'{prefix}{batch}-{suffix}.json' for batch in (5,6) for prefix,suffix in [('round','visual-reviews'),('lvis-round','downloads'),('round','duplicate-screen')]]

def candidates():
    manifest=m.MANIFEST;seen_shas={r['sha'] for r in manifest['records']};seen_pixels={r['pixels'] for r in manifest['records']}
    # Earlier external staged photos are conservatively protected too, including held sources.
    for name in ['duplicate-screen.json','round2-duplicate-screen.json','round3-duplicate-screen.json','round4-duplicate-screen.json']:
        for r in s.load(REVIEW/name)['records']:seen_shas.add(r['sha']);seen_pixels.add(r['pixels'])
    result=[]
    for batch in (5,6):
        review,downloads,screen=[s.load(REVIEW/name) for name in (f'round{batch}-visual-reviews.json',f'lvis-round{batch}-downloads.json',f'round{batch}-duplicate-screen.json')]
        added=admission.admit(review,downloads,screen,s.sha(REVIEW/f'round{batch}-duplicate-screen.json'),seen_shas,seen_pixels)
        result.extend(added);seen_shas.update(r['image_sha'] for r in added);seen_pixels.update(r['pixels'] for r in added)
    return result

def verify():
    base.verify();added=candidates();audit=s.load(RUN/'dataset.json')
    assert audit['base_dataset_audit_sha']==s.sha(BASE/'dataset.json')
    assert audit['provenance_shas']=={name:s.sha(REVIEW/name) for name in PROVENANCE}
    exposures,_=base.held_exposures();existing_shas={s.sha(Path(p)) for p in exposures}
    for source,record in zip(added,audit['added'],strict=True):
        assert all(record[k]==v for k,v in source.items()) and record['image_sha'] not in existing_shas
        assert s.sha(Path(record['image']))==record['image_sha']
        assert Path(record['label']).read_text().splitlines()==record['rows']
        exposures[record['image']]+=4
    listing=(RUN/'training-exposures.txt').read_text().splitlines()
    assert Counter(listing)==exposures and s.sha(RUN/'training-exposures.txt')==audit['training_listing_sha']
    assert yaml.safe_load((RUN/'data.yaml').read_text())==audit['config']
    assert len(added)==audit['added_images'] and sum(len(r['rows']) for r in added)==audit['added_boxes']
    assert len(exposures)==audit['training_unique_images']==1114+len(added)
    assert len(listing)==audit['training_exposures']==1956+4*len(added)
    boxes=sum(len(experiment.label_path(Path(p)).read_text().splitlines()) for p in exposures)
    assert boxes==audit['training_unique_bottle_boxes']==470+audit['added_boxes']
    assert audit['validation_images']==246 and audit['validation_bottle_boxes']==67
    return audit

if __name__=='__main__':
    source=base.verify();added=candidates();exposures,holds=base.held_exposures()
    for r in added:
        image=CACHE/'images/train'/Path(r['source_image']).name;label=CACHE/'labels/train'/f'{image.stem}.txt'
        image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
        if image.exists():assert s.sha(image)==r['image_sha']
        else:shutil.copyfile(r['source_image'],image)
        text='\n'.join(r['rows'])+'\n'
        if label.exists():assert label.read_text()==text
        else:label.write_text(text)
        r.update(image=str(image),label=str(label));exposures[str(image)]+=4
    listing=[p for p,count in exposures.items() for _ in range(count)];p=RUN/'training-exposures.txt';p.write_text('\n'.join(listing)+'\n')
    config=dict(path=str(m.DATA),train=str(p),val=str(m.DATA/'images/val'),names=['bottle']);(RUN/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    audit=dict(base_dataset_audit_sha=s.sha(BASE/'dataset.json'),fixed_source_fingerprint=source['fixed_source_fingerprint'],validation_protocol=source['validation_protocol'],provenance_shas={name:s.sha(REVIEW/name) for name in PROVENANCE},added=added,added_images=len(added),added_boxes=sum(len(r['rows']) for r in added),training_unique_images=len(exposures),training_unique_bottle_boxes=470+sum(len(r['rows']) for r in added),training_exposures=len(listing),training_listing_sha=s.sha(p),config=config,validation_images=246,validation_bottle_boxes=67,cropped_exposures=0,training_quality_holds=source['training_quality_holds'],human_verified=False,independent_test=False,review='AI full-scene and target-crop reviews plus independent second AI review; overlap screen heuristic')
    s.save(RUN/'dataset.json',audit);verify();print('V31_REVIEWED_FULL_IMAGE_DATA',len(exposures),audit['training_unique_bottle_boxes'],len(listing),flush=True)
