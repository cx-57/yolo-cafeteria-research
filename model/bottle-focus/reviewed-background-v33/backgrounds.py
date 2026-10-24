"""Keep positive exposures and replace unknown backgrounds with reviewed absence photos."""
from collections import Counter
from pathlib import Path
import hashlib
import random


def label_path(image):
    image=Path(image);return image.parents[2]/'labels/train'/f'{image.stem}.txt'

def resample(exposures,reviews,negative_budget,seed=43):
    assert type(negative_budget) is int and negative_budget>0
    exposures=Counter(exposures);assert all(type(n) is int and n>0 for n in exposures.values())
    negatives={p for p in exposures if not label_path(p).read_text().strip()}
    assert sum(exposures[p] for p in negatives)==negative_budget
    assert len({r['image'] for r in reviews})==len(reviews)
    accepted=[]
    for r in reviews:
        path=r['image'];assert path in negatives
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==r['image_sha']
        assert hashlib.sha256(label_path(path).read_bytes()).hexdigest()==r['label_sha']
        if not r['accepted']:continue
        assert r['full_scene_reviewed'] is True and r['independent_second_review'] is True and r['human_verified'] is False
        accepted.append(path)
    if not accepted:raise ValueError('No reviewed absence photos')
    accepted.sort();random.Random(seed).shuffle(accepted)
    assert len(accepted)<=negative_budget
    quotient,remainder=divmod(negative_budget,len(accepted))
    result=Counter({p:n for p,n in exposures.items() if p not in negatives})
    for index,path in enumerate(accepted):result[path]=quotient+(index<remainder)
    assert sum(result[p] for p in negatives)==negative_budget
    return result
