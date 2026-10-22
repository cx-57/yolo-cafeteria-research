"""Archive verified phase results and retain the best exact model bytes."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

RUN = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, required=True)
parser.add_argument('--run', type=Path, default=RUN)
args = parser.parse_args()
RUN = args.run.resolve()
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
incumbent = RUN.parent / 'verified-incumbent'
incumbent.mkdir(exist_ok=True)
for tag in ['selected', 'selected-tta640', 'averaged', 'averaged-tta640']:
    report_path = RUN / (tag + '.json')
    if not report_path.exists():
        continue
    report = json.loads(report_path.read_text())
    source = Path(report['weights'])
    assert sha(source) == report['weights_sha']
    destination = RUN / f'phase{args.epoch}-{source.name}'
    if destination.exists():
        assert sha(destination) == report['weights_sha']
    else:
        shutil.copy2(source, destination)
    report.update(weights=str(destination), snapshot_epoch=args.epoch,
                  source_verification=str(report_path))
    (RUN / f'phase{args.epoch}-{tag}.json').write_text(json.dumps(report, indent=2) + '\n')
    current = json.loads((incumbent / 'verification.json').read_text())
    if report['AP50'] > current['AP50']:
        shutil.copy2(destination, incumbent / 'bottle.pt')
        cached = Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/weights/bottle-best-verified.pt')
        shutil.copy2(destination, cached)
        assert sha(cached) == report['weights_sha'] == sha(incumbent / 'bottle.pt')
        report.update(source_checkpoint=str(destination), weights=str(incumbent / 'bottle.pt'),
                      runtime_cache_copy=str(cached))
        (incumbent / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
        print('NEW_VERIFIED_INCUMBENT', report['AP50'], report['weights_sha'])
if (RUN / 'averaging.json').exists():
    shutil.copy2(RUN / 'averaging.json', RUN / f'phase{args.epoch}-averaging.json')
print('Archived phase', args.epoch)
