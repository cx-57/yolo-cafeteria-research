"""Replace a bounded share of repeated training views with label-preserving focus crops."""
from pathlib import Path
from collections import Counter
import importlib.util
import yaml
from PIL import Image
import experiment

RUN=Path(__file__).resolve().parent
BASE=RUN.parent/'reviewed-expansion-v28'
CACHE=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-focus-crops-v30')
spec=importlib.util.spec_from_file_location('v30_base',BASE/'build.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
m=base.m;s=base.s

def held_exposures():
    exposures=Counter((BASE/'training-exposures.txt').read_text().splitlines())
    holds=s.load(RUN/'training-quality-holds.json');assert len(holds)==1
    for r in holds:
        path=Path(r['image']);label=experiment.label_path(path)
        assert path.name==r['name'] and s.sha(path)==r['image_sha'] and s.sha(label)==r['label_sha']
        assert label.read_text().splitlines()==r['rows'] and len(r['rows'])==1
        assert exposures[str(path)]==8
        del exposures[str(path)]
    return exposures,holds

def verify():
    source=base.verify();audit=s.load(RUN/'dataset.json')
    assert audit['base_dataset_audit_sha']==s.sha(BASE/'dataset.json')
    assert audit['crop_helper_sha']==s.sha(Path(experiment.__file__))
    listing=(RUN/'training-exposures.txt').read_text().splitlines()
    expected,holds=held_exposures()
    assert audit['training_quality_hold_sha']==s.sha(RUN/'training-quality-holds.json')
    for r in audit['crops']:
        original=Path(r['source_image']);label=experiment.label_path(original)
        assert s.sha(original)==r['source_image_sha'] and s.sha(label)==r['source_label_sha']
        assert s.sha(Path(r['image']))==r['image_sha']
        assert Path(r['label']).read_text().splitlines()==r['rows']
        expected[str(original)]-=1;expected[r['image']]+=1
    assert expected==Counter(listing) and len(listing)==1956
    assert all(count>0 for count in expected.values())
    assert yaml.safe_load((RUN/'data.yaml').read_text())==audit['config']
    assert s.sha(RUN/'training-exposures.txt')==audit['training_listing_sha']
    assert audit['validation_images']==246 and audit['validation_bottle_boxes']==67
    assert audit['original_training_photos']==1114
    assert audit['original_training_bottle_boxes']==470
    assert audit['training_unique_images']==len(expected)
    assert audit['training_unique_bottle_boxes']==470+sum(len(r['rows']) for r in audit['crops'])
    return audit

if __name__=='__main__':
    source=base.verify();exposures,holds=held_exposures();crops=[]
    for name,count in list(exposures.items()):
        if count<4:continue
        original=Path(name);lab=experiment.label_path(original);rows=lab.read_text().splitlines()
        with Image.open(original) as im:assert im.getexif().get(274,1)==1
        for view in range(1,count//4+1):
            result=experiment.focus_crop(original,rows,['bottle'],['bottle'],view)
            if result is None:continue
            picture,clipped=result
            image=CACHE/'images/train'/f'{original.stem}__focus{view}.jpg';label=experiment.label_path(image)
            image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
            picture.save(image,quality=95);clipped=experiment.validate_rows(clipped,image,['bottle'])
            label.write_text('\n'.join(clipped)+'\n')
            crops.append(dict(source_image=name,source_image_sha=s.sha(original),source_label_sha=s.sha(lab),view=view,image=str(image),label=str(label),image_sha=s.sha(image),rows=clipped))
            exposures[name]-=1;exposures[str(image)]+=1
    assert len(crops)==187 and len({r['source_image'] for r in crops})==102
    listing=[p for p,count in exposures.items() for _ in range(count)]
    p=RUN/'training-exposures.txt';p.write_text('\n'.join(listing)+'\n')
    config=dict(path=str(m.DATA),train=str(p),val=str(m.DATA/'images/val'),names=['bottle'])
    (RUN/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    audit=dict(source,base_dataset_audit_sha=s.sha(BASE/'dataset.json'),crop_helper_sha=s.sha(Path(experiment.__file__)),training_quality_hold_sha=s.sha(RUN/'training-quality-holds.json'),training_quality_holds=source['training_quality_holds']+holds,context_positive_photos=113,crops=crops,cropped_exposures=187,crop_source_photos=102,original_training_photos=1114,original_training_bottle_boxes=470,training_unique_images=len(set(listing)),training_unique_bottle_boxes=470+sum(len(r['rows']) for r in crops),training_exposures=1956,training_listing_sha=s.sha(p),config=config,crop_fraction=.6,crop_target_max_area=.04,partial_label_policy='Existing focus_crop clips every intersecting annotation and omits subpixel intersections; no source annotation is changed',independent_test=False)
    s.save(RUN/'dataset.json',audit);verify();print('FOCUS_CROP_DATA_VERIFIED',1114,1956,len(crops),flush=True)
