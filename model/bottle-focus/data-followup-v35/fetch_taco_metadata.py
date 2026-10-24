"""Read per-photo Flickr attribution/license metadata; no training admission."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
import subprocess

RUN = Path(__file__).resolve().parent

def read(record):
    photo_id = Path(record['original']['flickr_url']).name.split('_')[0]
    url = 'https://www.flickr.com/photo.gne?id=' + photo_id
    try:
        raw = subprocess.run(['curl', '-fsSL', '--max-time', '25', url], check=True, capture_output=True).stdout
        graphs = [json.loads(text) for text in re.findall(r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', raw.decode(), re.S)]
        objects = [node for graph in graphs for node in graph.get('@graph', [graph]) if node.get('@type') == 'ImageObject']
        assert len(objects) == 1
        obj = objects[0]
        assert Path(obj['contentUrl']).name.split('_')[0] == photo_id
        return dict(id=record['id'], photo_id=photo_id, source_url=url, source_html_sha=hashlib.sha256(raw).hexdigest(),
                    photo_page=obj.get('acquireLicensePage'), license_url=obj.get('license'), author=obj.get('author'),
                    content_url=obj['contentUrl'], retrieved_at=datetime.now(timezone.utc).isoformat(),
                    status='source_metadata_only_no_training_admission')
    except Exception as exc:
        return dict(id=record['id'], photo_id=photo_id, source_url=url, status='metadata_hold', error=str(exc))

if __name__ == '__main__':
    records = json.loads((RUN/'taco-downloads.json').read_text())
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = []
        for row in pool.map(read, records):
            rows.append(row)
            print('PHOTO_METADATA', len(rows), row['id'], row['status'], row.get('license_url'), flush=True)
    (RUN/'taco-photo-metadata.json').write_text(json.dumps(rows, indent=2)+'\n')
