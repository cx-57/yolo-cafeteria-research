#!/usr/bin/env bash
set -euo pipefail

# Upload both reviewed food sets without modifying older source labels.
cd "$(git rev-parse --show-toplevel)"
[[ "$(git remote get-url origin)" == *"cx-57/yolo-cafeteria-research"* ]] || { echo "Wrong repository" >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "Commit/stash tracked edits first" >&2; exit 1; }
git switch main
git pull --ff-only origin main
ZIP="${1:-$HOME/Downloads/NHS_Food_Additions_510.zip}"
[[ -f "$ZIP" ]] || { echo "Missing ZIP: $ZIP" >&2; exit 1; }

python3 - "$ZIP" <<'PY'
import csv, hashlib, io, math, sys
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

expected = {'school_lunch_asian': 130, 'food_american': 380}
root = Path('data/processed')
manifests = Path('data/manifests')
with ZipFile(Path(sys.argv[1]).expanduser()) as z:
    if z.testzip() is not None:
        sys.exit('Corrupted ZIP')
    rows = list(csv.DictReader(io.StringIO(z.read('manifest.csv').decode('utf-8-sig'))))
    names = set(z.namelist())
    counts, classes, staged = Counter(), Counter(), {}
    for row in rows:
        source, name = row['source'], row['processed_image']
        if source not in expected or Path(name).name != name or not name.endswith('.jpg'):
            sys.exit('Unexpected source/filename in manifest')
        img, label = f'{source}/images/{name}', f'{source}/labels/{Path(name).stem}.txt'
        if img not in names or label not in names:
            sys.exit(f'Missing image or label: {name}')
        original = z.read(img)
        if hashlib.sha256(original).hexdigest() != row['sha256'] or not original.startswith(b'\xff\xd8'):
            sys.exit(f'Invalid image/hash: {name}')
        raw_labels = z.read(label)
        lines = raw_labels.decode().strip().splitlines()
        if not lines or len(lines) != int(row['instances']):
            sys.exit(f'Wrong box count: {label}')
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                sys.exit(f'Not a detection box: {label}')
            cid = int(parts[0]); x,y,w,h = map(float, parts[1:])
            if (cid not in (0,6) or (source == 'food_american' and cid != 0)
                    or not all(map(math.isfinite,(x,y,w,h))) or w <= 0 or h <= 0
                    or x-w/2 < -1e-5 or x+w/2 > 1+1e-5 or y-h/2 < -1e-5 or y+h/2 > 1+1e-5):
                sys.exit(f'Bad box: {label}')
            classes[cid] += 1
        counts[source] += 1
        staged[str(root/source/'images'/name)] = original
        staged[str(root/source/'labels'/(Path(name).stem+'.txt'))] = raw_labels
    if counts != expected or len(staged) != 1020 or classes != {0:1072,6:125}:
        sys.exit(f'Unexpected counts: {counts}, {classes}')
    if len([n for n in names if '/images/' in n and n.endswith('.jpg')]) != 510:
        sys.exit('Extra photos in ZIP')
    for source in expected:
        buf = io.StringIO(newline='')
        writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(r for r in rows if r['source']==source)
        staged[str(manifests/(source+'.csv'))] = buf.getvalue().encode()
        for folder in ('images','labels'):
            directory = root/source/folder
            if directory.is_dir():
                extra = {str(p) for p in directory.iterdir() if p.is_file()} - set(staged)
                if extra:
                    sys.exit(f'Unexpected existing files in {directory}')
    for filename, payload in staged.items():
        path = Path(filename)
        if path.exists() and path.read_bytes() != payload:
            sys.exit(f'Conflicting existing file: {filename}')
    for filename, payload in staged.items():
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(payload)
print('Verified 510 image/label pairs and 1,197 boxes: 130 Asian, 380 American.')
print('All added photos are training-only; original source labels remain unchanged.')
PY

git add data/processed/school_lunch_asian data/processed/food_american data/manifests/school_lunch_asian.csv data/manifests/food_american.csv
if ! git diff --cached --quiet; then
  git commit -m "Add 510 reviewed Asian and American cafeteria food photos"
fi
git push origin main
echo "Finished. Run: python3 src/train.py --build-only"
