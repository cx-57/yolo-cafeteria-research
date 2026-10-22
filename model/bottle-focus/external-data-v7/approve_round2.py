"""Record AI visual review decisions, bound to exact downloaded source bytes."""
from pathlib import Path
import hashlib,json
RUN=Path(__file__).resolve().parent
records=[r for r in json.loads((RUN/'lvis-round2-downloads.json').read_text()) if r['status']=='unreviewed']
accepted={i:None for i in [0,3,6,8,11,12,13,14,15,16,20,23,25,26,27,31]}
screen={r['id']:r for r in json.loads((RUN/'round2-duplicate-screen.json').read_text())['records']}
reviews=[]
for index,r in enumerate(records):
    sha=hashlib.sha256(Path(r['image']).read_bytes()).hexdigest()
    assert sha==r['sha']
    chosen=list(range(len(r['annotations']))) if accepted.get(index) is None else accepted[index]
    admit=index in accepted and not screen[r['id']]['hold']
    reason='AI visual review: visible plastic beverage bottles; source boxes retained; no additional qualifying bottle identified.' if admit else 'Not admitted: material, annotation completeness, tiny-target visibility or scene suitability unresolved.'
    reviews.append(dict(id=r['id'],index=index,image_sha=sha,accepted=admit,annotation_indices=chosen if admit else [],reviewer='Codex AI visual review',human_verified=False,scope='Experimental training only; never validation/test',reason=reason,source_annotations=r['annotations']))
(RUN/'round2-visual-reviews.json').write_text(json.dumps(reviews,indent=2)+'\n')
print('ROUND2_REVIEWED',len(reviews),'ADMITTED',sum(r['accepted'] for r in reviews),'BOTTLE_BOXES',sum(len(r['annotation_indices']) for r in reviews if r['accepted']))
