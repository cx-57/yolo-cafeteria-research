"""Build reviewed absence backgrounds and whole-photo Open Images positives, training only."""
from pathlib import Path
from collections import Counter
import importlib.util
import shutil
import yaml

RUN=Path(__file__).resolve().parent
BASE=RUN.parent/'reviewed-neck-v31'
OI=RUN.parent/'openimages-staging-v32'
CACHE=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-reviewed-background-v33')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
base=module('v33_base',BASE/'build.py');m=base.m;s=base.s
admission=module('v33_admission',RUN/'admission.py');backgrounds=module('v33_backgrounds',RUN/'backgrounds.py')
PROVENANCE=['targeted-downloads.json','targeted-visual-reviews.json','targeted-duplicate-screen.json','targeted-annotation-prefix.json','targeted-photo-metadata.json','plastic-water-image-labels.json','flickr-identity-screen.json']

def base_exposures():
    return Counter((BASE/'training-exposures.txt').read_text().splitlines())

def reviewed_exposures():
    reviews=s.load(RUN/'negative-reviews.json')
    candidates=s.load(RUN/'negative-review-candidates.json')+s.load(RUN/'additional-negative-review-candidates.json')
    assert {r['image'] for r in candidates}=={r['image'] for r in reviews} and len(candidates)==len(reviews)==180
    for candidate,review in zip(candidates,reviews,strict=True):
        assert all(review[k]==v for k,v in candidate.items())
    result=backgrounds.resample(base_exposures(),reviews,749,seed=43)
    assert sum(r['accepted'] for r in reviews)==166
    return result

def candidates():
    seen_shas={r['sha'] for r in m.MANIFEST['records']};seen_pixels={r['pixels'] for r in m.MANIFEST['records']}
    external=RUN.parent/'external-data-v7'
    for name in ['duplicate-screen.json']+[f'round{i}-duplicate-screen.json' for i in range(2,7)]:
        for r in s.load(external/name)['records']:seen_shas.add(r['sha']);seen_pixels.add(r['pixels'])
    downloads,review,screen,groups,metadata=[s.load(OI/name) for name in PROVENANCE[:5]]
    positive_labels=s.load(OI/'plastic-water-image-labels.json');label_by_id={r['id']:r['positive_source_labels'] for r in positive_labels['records']}
    assert groups['image_label_filter_sha256']==s.sha(OI/'plastic-water-image-labels.json')
    assert all(r['positive_source_labels']==label_by_id[r['id']] for r in groups['groups'])
    identities=s.load(OI/'flickr-identity-screen.json')
    assert identities['lvis_index_sha256']==s.sha(Path(identities['lvis_index']))
    held={r['id'] for r in identities['records'] if r['hold']}
    assert not any(r['accepted'] and r['id'] in held for r in review)
    added=admission.admit(review,downloads,screen,s.sha(OI/'targeted-duplicate-screen.json'),groups['groups'],metadata['found'],seen_shas,seen_pixels)
    assert len(added)==23 and sum(len(r['rows']) for r in added)==34
    return added

def verify():
    base.verify();audit=s.load(RUN/'dataset.json');added=candidates();exposures=reviewed_exposures()
    assert audit['base_dataset_audit_sha']==s.sha(BASE/'dataset.json')
    assert audit['provenance_shas']=={name:s.sha(OI/name) for name in PROVENANCE}
    assert audit['negative_review_sha']==s.sha(RUN/'negative-reviews.json')
    assert audit['negative_candidate_shas']=={name:s.sha(RUN/name) for name in ['negative-review-candidates.json','additional-negative-review-candidates.json']}
    original=base_exposures();prior_positive={p:n for p,n in original.items() if backgrounds.label_path(p).read_text().strip()}
    assert {p:exposures[p] for p in prior_positive}==prior_positive
    existing_shas={s.sha(Path(p)) for p in original}
    for source,record in zip(added,audit['added'],strict=True):
        assert all(record[k]==v for k,v in source.items()) and record['image_sha'] not in existing_shas
        assert s.sha(Path(record['image']))==record['image_sha']
        assert Path(record['label']).read_text().splitlines()==record['rows']
        exposures[record['image']]+=4
    listing=(RUN/'training-exposures.txt').read_text().splitlines()
    assert Counter(listing)==exposures and s.sha(RUN/'training-exposures.txt')==audit['training_listing_sha']
    assert yaml.safe_load((RUN/'data.yaml').read_text())==audit['config']
    negative=[p for p in exposures if not backgrounds.label_path(p).read_text().strip()]
    boxes=sum(len(backgrounds.label_path(p).read_text().splitlines()) for p in exposures)
    assert len(exposures)==audit['training_unique_images']==579
    assert len(listing)==audit['training_exposures']==2148
    assert boxes==audit['training_unique_bottle_boxes']==538
    assert len(negative)==audit['reviewed_negative_photos']==166 and sum(exposures[p] for p in negative)==749
    assert audit['validation_images']==246 and audit['validation_bottle_boxes']==67
    return audit

if __name__=='__main__':
    source=base.verify();added=candidates();exposures=reviewed_exposures()
    for r in added:
        image=CACHE/'images/train'/Path(r['source_image']).name;label=backgrounds.label_path(image)
        image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
        if image.exists():assert s.sha(image)==r['image_sha']
        else:shutil.copyfile(r['source_image'],image)
        text='\n'.join(r['rows'])+'\n'
        if label.exists():assert label.read_text()==text
        else:label.write_text(text)
        r.update(image=str(image),label=str(label));exposures[str(image)]+=4
    listing=[p for p,count in exposures.items() for _ in range(count)];p=RUN/'training-exposures.txt';p.write_text('\n'.join(listing)+'\n')
    config=dict(path=str(m.DATA),train=str(p),val=str(m.DATA/'images/val'),names=['bottle']);(RUN/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    audit=dict(base_dataset_audit_sha=s.sha(BASE/'dataset.json'),fixed_source_fingerprint=source['fixed_source_fingerprint'],validation_protocol=source['validation_protocol'],provenance_shas={name:s.sha(OI/name) for name in PROVENANCE},negative_review_sha=s.sha(RUN/'negative-reviews.json'),negative_candidate_shas={name:s.sha(RUN/name) for name in ['negative-review-candidates.json','additional-negative-review-candidates.json']},added=added,added_images=23,added_boxes=34,training_unique_images=len(exposures),training_unique_bottle_boxes=538,training_exposures=len(listing),training_listing_sha=s.sha(p),config=config,reviewed_negative_photos=166,negative_exposures=749,unreviewed_or_uncertain_negative_photos_held=583,prior_positive_photos=390,prior_positive_exposures=1307,new_positive_exposures=92,validation_images=246,validation_bottle_boxes=67,training_quality_holds_inherited=dict(base_dataset=str(BASE/'dataset.json'),base_audit_sha=s.sha(BASE/'dataset.json')),human_verified=False,independent_test=False,review='AI full-scene/crop reviews plus independent second AI review, not human certification; overlap screening is heuristic; reviewed background repeats preserve aggregate negative exposure volume but change its diversity')
    s.save(RUN/'dataset.json',audit);verify();print('V33_REVIEWED_BACKGROUND_DATA_VERIFIED',len(exposures),538,len(listing),flush=True)
