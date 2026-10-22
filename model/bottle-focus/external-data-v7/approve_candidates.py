"""Record AI visual review decisions, bound to exact downloaded source bytes."""
from pathlib import Path
import hashlib,json
RUN=Path(__file__).resolve().parent
records=[r for r in json.loads((RUN/'lvis-downloads.json').read_text()) if r['status']=='unreviewed']
accepted={0:None,1:None,2:None,3:None,7:None,9:None,12:None,13:None,17:[0],20:None,22:None,26:None,27:None,29:None,31:None,33:None,34:None,35:None,36:None}
screen={r['id']:r for r in json.loads((RUN/'duplicate-screen.json').read_text())['records']}
reviews=[]
for index,r in enumerate(records):
    sha=hashlib.sha256(Path(r['image']).read_bytes()).hexdigest()
    assert sha==r['sha']
    chosen=list(range(len(r['annotations']))) if accepted.get(index) is None else accepted[index]
    admit=index in accepted and not screen[r['id']]['hold']
    reason='AI visual review: visible plastic beverage bottles; source boxes retained; no additional qualifying bottle identified.' if admit else 'Not admitted: material, annotation completeness, tiny-target visibility or scene suitability unresolved.'
    if index==17:reason+=' Excluded annotation 1: reusable insulated bottle; retained PET water bottle annotation 0.'
    if index==19:reason='Held: birthday scene visually related to fixed validation photos, despite passing heuristic feature screen.'
    reviews.append(dict(id=r['id'],index=index,image_sha=sha,accepted=admit,annotation_indices=chosen if admit else [],reviewer='Codex AI visual review',human_verified=False,scope='Experimental training only; never validation/test',reason=reason,source_annotations=r['annotations']))
(RUN/'visual-reviews.json').write_text(json.dumps(reviews,indent=2)+'\n')
(RUN/'source-rejection.json').write_text(json.dumps(dict(source='https://data.mendeley.com/datasets/tksn2jrckf/1',reviewed_sample=24,admitted=0,reason='Repeated shop scenes and many unlabeled visible bottles; out-of-bounds annotations in sample images 10 and 23. Native EXIF display orientation was checked. Dataset not admitted as supplied.'),indent=2)+'\n')
print('REVIEWED',len(reviews),'ADMITTED',sum(r['accepted'] for r in reviews),'BOTTLE_BOXES',sum(len(r['annotation_indices']) for r in reviews if r['accepted']))
