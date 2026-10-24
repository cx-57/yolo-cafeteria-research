"""Conservative staged-photo overlap screen; never admits data or changes training."""
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
from PIL import Image, ImageOps
from audit import difference_hash

RUN = Path(__file__).resolve().parent
BASE = Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-capture-validation')
cv2.setNumThreads(1)
SIFT = cv2.SIFT_create(nfeatures=350)
MATCHER = cv2.BFMatcher()

def features(path):
    raw = Path(path).read_bytes()
    with Image.open(path) as source:
        im = ImageOps.exif_transpose(source).convert('RGB')
    pixel = hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
    dh = [difference_hash(im.rotate(k*90, expand=True)) for k in range(4)]
    im.thumbnail((640,640))
    points, desc = SIFT.detectAndCompute(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY), None)
    return dict(sha=hashlib.sha256(raw).hexdigest(), pixels=pixel, dh=dh, points=points, desc=desc)

def compare(candidate, targets):
    byte_matches, pixel_matches, dhash_matches, feature_matches = [], [], [], []
    for name, other in targets:
        if candidate['sha'] == other['sha']:
            byte_matches.append(name)
        if candidate['pixels'] == other['pixels']:
            pixel_matches.append(name)
        distance = min((a^b).bit_count() for a in candidate['dh'] for b in other['dh'])
        if distance <= 8:
            dhash_matches.append(dict(name=name, distance=distance))
        a, b = candidate['desc'], other['desc']
        if a is None or b is None or len(a)<2 or len(b)<2:
            continue
        def ratio(first, second):
            return {x.queryIdx:x.trainIdx for pair in MATCHER.knnMatch(first, second, k=2)
                    if len(pair)==2 for x,y in [pair] if x.distance < .7*y.distance}
        forward, reverse = ratio(a,b), ratio(b,a)
        mutual = [(x,y) for x,y in forward.items() if reverse.get(y)==x]
        if len(mutual)<6:
            continue
        source = np.float32([candidate['points'][x].pt for x,y in mutual])
        destination = np.float32([other['points'][y].pt for x,y in mutual])
        _, mask = cv2.findHomography(source, destination, cv2.RANSAC, 4.)
        inliers = int(mask.sum()) if mask is not None else 0
        if inliers>=6:
            feature_matches.append(dict(name=name, mutual=len(mutual), inliers=inliers))
    return dict(byte_matches=byte_matches, pixel_matches=pixel_matches, dhash_matches=dhash_matches,
                feature_matches=feature_matches, hold=bool(byte_matches or pixel_matches or dhash_matches or feature_matches))

def load(path):
    return json.loads(path.read_text())

def snapshot(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()

def check_inputs(identities):
    for path, expected in identities.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
            raise ValueError('Input changed: '+str(path))

def protected_features(path, expected):
    result = features(path)
    if result['sha']!=expected:
        raise ValueError('Protected image changed: '+str(path))
    return result

def main():
    identities = {}
    def bound(path):
        value, digest = snapshot(path)
        identities[str(path)] = digest
        return value
    inputs = ['additional-downloads.json','positive-second-review.json','negative-candidates.json',
              'negative-second-review.json','taco-downloads.json','taco-second-review.json','taco-photo-metadata.json']
    contents = {name:bound(RUN/name) for name in inputs}
    identities[str(Path(__file__))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = bound(BASE/'manifest.json')
    expected_by_name = {r['name']:r['sha'] for r in manifest['records']}
    # Historical added-photo audits provide expected identities for current training aliases.
    for audit_path in sorted(RUN.parent.glob('*/dataset.json')):
        if audit_path.parent==RUN:
            continue
        audit = bound(audit_path)
        for row in audit.get('added',[]):
            path, digest = row.get('image'), row.get('image_sha')
            if path and digest:
                name = Path(path).name
                if name in expected_by_name:
                    assert expected_by_name[name]==digest, 'Conflicting historical photo identity'
                expected_by_name[name] = digest
    protected = []
    for row in manifest['records']:
        protected.append((('benchmark/' if row['split']=='val' else 'historical-training/')+row['name'], BASE/'images'/row['split']/row['name'], row['sha']))
    active = RUN.parent/'monitor-campaign-v34'
    active_audit = bound(active/'dataset.json')
    assert len(active_audit['monitor_records'])==36
    assert sum(r['split']=='val' for r in manifest['records'])==246
    for row in active_audit['monitor_records']:
        protected.append(('monitor/'+row['id'], Path(row['image']), row['image_sha']))
    listing = active/'training-exposures.txt'
    raw = listing.read_bytes()
    identities[str(listing)] = hashlib.sha256(raw).hexdigest()
    assert identities[str(listing)]==active_audit['training_listing_sha']
    for path in sorted(set(raw.decode().splitlines())):
        expected = expected_by_name[Path(path).name]
        protected.append(('current-training/'+Path(path).name, Path(path), expected))
    prepared, seen, protected_records = [], set(), []
    for name,path,expected in protected:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=expected:
            raise ValueError('Protected image changed: '+str(path))
        identities[str(path)] = digest
        protected_records.append(dict(role_name=name, image=str(path), sha=digest))
        if digest not in seen:
            seen.add(digest)
            prepared.append((name, protected_features(path, expected)))
    print('PROTECTED_PHOTOS', len(prepared), flush=True)
    oi = contents['additional-downloads.json']
    positives = contents['positive-second-review.json']['accepted_indices']
    generic = contents['negative-candidates.json']
    negatives = set(contents['negative-second-review.json']['accepted_indices'])
    # Generic staging rows retain their original download index in a separate field.
    candidates = [dict(oi[i], role='positive', source='openimages') for i in positives]
    for row in generic:
        if row['index'] in negatives:
            candidates.append(dict(row, sha=row['image_sha'], role='negative', source='openimages'))
    taco = contents['taco-downloads.json']
    metadata = {r['id']:r for r in contents['taco-photo-metadata.json']}
    for i in contents['taco-second-review.json']['accepted_indices']:
        row = taco[i]
        if metadata[row['id']].get('license_url','').startswith('https://creativecommons.org/licenses/'):
            candidates.append(dict(row, role='positive', source='taco'))
    results = []
    for row in candidates:
        f = features(row['image'])
        assert f['sha']==row['sha']
        identities[str(Path(row['image']))] = f['sha']
        result = dict(id=row['id'], source=row['source'], role=row['role'], image=row['image'],
                      sha=f['sha'], pixels=f['pixels'], **compare(f, prepared))
        results.append(result)
        # Every candidate joins the comparison pool, including held ones; conservatively group related captures.
        prepared.append(('candidate/'+row['id'], f))
        print('FOLLOWUP_OVERLAP',row['id'],'HOLD' if result['hold'] else 'PASS',flush=True)
    check_inputs(identities)
    output = dict(status='screened_only_no_training_admission', inputs=identities, protected_records=protected_records,
                  protected_unique_photos=len(seen), benchmark_images=246, permanent_monitor_images=36,
                  records=results, limitations='Byte/pixel plus rotation-aware dHash<=8 and SIFT350 mutual ratio0.7/RANSAC4px>=6inliers. Heuristic screen; not proof of independent sources. All flags held; training unchanged.')
    (RUN/'overlap-screen.json').write_text(json.dumps(output,indent=2)+'\n')

if __name__=='__main__':
    main()
