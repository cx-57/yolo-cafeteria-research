"""Stage unused official TACO photos; no training admission or active-run edits."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import random
import subprocess
from PIL import Image, ImageDraw, ImageOps

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
SOURCE = ROOT / 'data/focus-three-class-improvement/bottle-candidates'
CACHE = Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-taco-staging-v35')

def fetch(record):
    image = record['original']
    target = CACHE / f"taco_{image['id']}.image"
    try:
        if not target.exists():
            partial = target.with_suffix('.partial')
            subprocess.run(['curl', '-fsSL', '--retry', '1', '--max-time', '40', image['flickr_url'], '-o', str(partial)], check=True, capture_output=True)
            partial.replace(target)
        with Image.open(target) as im:
            im.load()
            dimensions = im.size
            orientation = im.getexif().get(274, 1)
            suffix = { 'JPEG': '.jpg', 'PNG': '.png' }.get(im.format)
        if not suffix or dimensions != (image['width'], image['height']) or orientation != 1:
            return dict(record, status='geometry_or_orientation_hold', dimensions=dimensions, orientation=orientation)
        destination = target.with_suffix(suffix)
        target.replace(destination)
        return dict(record, image=str(destination), sha=hashlib.sha256(destination.read_bytes()).hexdigest(), status='unreviewed_staged_only', training_admission=False)
    except Exception as exc:
        return dict(record, status='download_or_geometry_hold', error=str(exc), training_admission=False)

if __name__ == '__main__':
    CACHE.mkdir(exist_ok=True)
    raw = (SOURCE / 'taco_annotations.json').read_bytes()
    data = json.loads(raw)
    old = {r['id'] for r in json.loads((SOURCE / 'candidates.json').read_text()) if r['source'] == 'TACO'}
    annotations = {}
    for annotation in data['annotations']:
        annotations.setdefault(annotation['image_id'], []).append(annotation)
    candidates = []
    for im in data['images']:
        anns = annotations.get(im['id'], [])
        clear = [a for a in anns if a['category_id'] == 5]
        if clear and f"taco_{im['id']}" not in old:
            candidates.append(dict(id=f"taco_{im['id']}", original=im, annotations=clear, all_source_annotations=anns,
                                   source_annotation_sha=hashlib.sha256(raw).hexdigest(),
                                   source_url='https://github.com/pedropro/TACO',
                                   license_status='Source annotation has no per-image license; unresolved, no training admission'))
    random.Random(1043).shuffle(candidates)
    candidates = candidates[:60]
    rows = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for row in pool.map(fetch, candidates):
            rows.append(row)
            print('TACO_STAGED', len(rows), row['id'], row['status'], flush=True)
    (RUN / 'taco-downloads.json').write_text(json.dumps(rows, indent=2) + '\n')
    for start in range(0, len(rows), 6):
        sheet = Image.new('RGB', (1536, 1100), 'white')
        draw = ImageDraw.Draw(sheet)
        for j, row in enumerate(rows[start:start+6]):
            x, y = j % 3 * 512, j // 3 * 550
            if row['status'] == 'unreviewed_staged_only':
                with Image.open(row['image']) as source:
                    im = source.convert('RGB')
                boxes = ImageDraw.Draw(im)
                for a in row['annotations']:
                    xx, yy, w, h = a['bbox']
                    boxes.rectangle((xx, yy, xx+w, yy+h), outline='red', width=max(2, im.width//400))
                im = ImageOps.contain(im, (504, 495))
                sheet.paste(im, (x, y))
            draw.text((x+3, y+502), f"{start+j}: {row['id']} {row['status']}", fill='black')
        sheet.save(RUN / f'taco-preview-{start:03}.jpg', quality=92)
    print('STAGING_COMPLETE_NO_TRAINING_ADMISSION', flush=True)
