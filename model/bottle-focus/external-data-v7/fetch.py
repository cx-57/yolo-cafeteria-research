"""Download checksum-verified review candidates from a new CC BY 4.0 source."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import random
import subprocess

RUN = Path(__file__).resolve().parent
CACHE = Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-mendeley-20261022')
SOURCE = 'https://data.mendeley.com/datasets/tksn2jrckf/1'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def fetch(record, folder):
    path = CACHE / folder / record['filename']
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = record['content_details']['sha256_hash']
    if not path.exists() or sha(path) != expected:
        subprocess.run(['curl', '-fsSL', '--retry', '2', '--max-time', '60', record['content_details']['download_url'], '-o', str(path)], check=True)
    assert sha(path) == expected, path
    return path

def pair(record):
    image = fetch(record, 'images')
    label = fetch(LABELS[Path(record['filename']).stem], 'labels')
    return dict(name=image.name, image=str(image), label=str(label), image_sha=sha(image),
                label_sha=sha(label), source=SOURCE, license='CC BY 4.0',
                author='Saifullah Anik', doi='10.17632/tksn2jrckf.1', status='unreviewed')

LABELS = {Path(x['filename']).stem: x for x in json.loads((RUN / 'source-labels.json').read_text())}
if __name__ == '__main__':
    candidates = json.loads((RUN / 'source-images.json').read_text())
    random.Random(42).shuffle(candidates)
    candidates = candidates[:int(os.environ.get('BOTTLE_CANDIDATES', '24'))]
    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for record in pool.map(pair, candidates):
            records.append(record)
            print('DOWNLOADED', len(records), record['name'], flush=True)
    (RUN / 'downloads.json').write_text(json.dumps(records, indent=2) + '\n')
