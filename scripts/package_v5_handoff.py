"""Build the self-contained Colab notebook from the maintained V5 modules."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'experiments/yolo-v5-original-plus-v4/YOLO_V5_Six_Class_Unified_First.ipynb'


def build(output=OUTPUT):
    cells = []

    def cell(kind, source, hidden=False):
        value = dict(cell_type=kind, metadata={}, source=source.splitlines(keepends=True))
        if kind == 'code':
            value.update(execution_count=None, outputs=[])
        if hidden:
            value['metadata']['jupyter'] = {'source_hidden': True}
        cells.append(value)

    cell('markdown', '''# V5: remove bottle, improve the unified detector first

Use **Runtime → Change runtime type → GPU**. Start with the 40-epoch baseline. Download its report before committing to the full campaign.

Six classes: food, plate, utensil, hinged_container, drink_carton, wrapper. Annotations are **provisional**. This notebook contains the Python code; it needs no GitHub branch download. No training has been performed to produce this handoff.

Required inputs: the corrected complete `YOLO_V5_CommandCenter_B187_QC_selected_merged1572.zip` and the documented 28-image ambiguity exclusion ledger. The old audit320 ZIP is not substituted. Specialists are deferred until actual unified results and repeat/uncertainty analysis justify them.
''')
    cell('code', '''import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'ultralytics==8.4.175', 'PyYAML>=6.0', 'Pillow>=10.0', 'ImageHash==4.3.2'], check=True)
import torch
assert torch.cuda.is_available(), 'Enable a GPU runtime before training.'
print('GPU:', torch.cuda.get_device_name(0))
''')
    cell('markdown', '## 1. Save data and runs in a persistent folder\n\nThe default mounts your Google Drive so runtime resets do not erase checkpoints. Keep the same WORK path when reopening this notebook. If you turn persistence off, download results before ending the session.\n')
    cell('code', '''from pathlib import Path
USE_DRIVE = True
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    WORK = Path('/content/drive/MyDrive/cafeteria-v5-six-class')
else:
    WORK = Path('/content/cafeteria-v5-six-class')
WORK.mkdir(parents=True, exist_ok=True)
CODE = Path('/content/v5-six-class-code')
CODE.mkdir(exist_ok=True)
DATA = WORK / 'data'
RUNS = WORK / 'runs'
print('Persistent work directory:', WORK)
''')
    payload = {name: (ROOT / 'scripts' / name).read_text() for name in
               ('v5_dataset.py', 'v5_trials.py', 'v5_experiment.py')}
    cell('markdown', '## 2. Install the included experiment scripts\n\nThis collapsed cell contains the source snapshot for the CLI, dataset preparation and trial runner. It does not train anything.\n')
    cell('code', 'SOURCES = ' + repr(payload) + '''
for name, source in SOURCES.items():
    path = CODE / name
    if path.exists() and path.read_text() != source:
        raise RuntimeError(f'Different experiment code already exists at {path}; start a fresh runtime.')
    path.write_text(source)
CLI = CODE / 'v5_experiment.py'
def cli(*args):
    subprocess.run([sys.executable, str(CLI), *map(str, args)], check=True)
cli('plan')
''', hidden=True)
    cell('markdown', '## 3. Upload the corrected dataset if it is not already saved\n\nSelect the complete corrected ZIP. You can upload the exclusion CSV alongside it. The next cell also lists likely exclusion-ledger files inside the ZIP; inspect them before mapping any nonstandard columns.\n')
    cell('code', '''import shutil, zipfile, json
from google.colab import files
ARCHIVE_NAME = 'YOLO_V5_CommandCenter_B187_QC_selected_merged1572.zip'
ARCHIVE = WORK / ARCHIVE_NAME
EXCLUSIONS = WORK / 'exclusions_28.csv'
if not ARCHIVE.is_file():
    uploaded = files.upload()
    del uploaded
    source = Path('/content') / ARCHIVE_NAME
    assert source.is_file(), f'Upload the exact corrected ZIP: {ARCHIVE_NAME}'
    shutil.copy2(source, ARCHIVE)
cli('audit', '--archive', ARCHIVE)
with zipfile.ZipFile(ARCHIVE) as z:
    ledger_candidates = [n for n in z.namelist() if n.lower().endswith('.csv')
                         and any(k in n.lower() for k in ('exclude', 'exclusion', 'uncert', 'decision'))]
    print('Possible QC/exclusion ledgers:', *ledger_candidates, sep='\\n')
''')
    cell('markdown', '''## 4. Provide the 28-image exclusion ledger

The ledger must contain an `image_path` column with exactly 28 unique paths matching the ZIP's manifest (for example `images/train/name.jpg`). If the ZIP includes that exact ledger, set `LEDGER_MEMBER` below to its listed path. Otherwise upload a mapped CSV. A JSON list of exact paths is also supported by the CLI. Do not guess identities from row numbers or exclude arbitrary photos. Preparation refuses a missing or invalid list.
''')
    cell('code', '''LEDGER_MEMBER = ''  # Set this only if the ZIP contains the exact, canonical 28-image CSV.
if not EXCLUSIONS.is_file():
    if LEDGER_MEMBER:
        with zipfile.ZipFile(ARCHIVE) as z:
            EXCLUSIONS.write_bytes(z.read(LEDGER_MEMBER))
    else:
        print('Upload the 28-image CSV with an image_path column.')
        uploaded = files.upload()
        csv_names = [n for n in uploaded if n.lower().endswith('.csv')]
        assert len(csv_names) == 1, 'Upload exactly one mapped exclusion CSV.'
        EXCLUSIONS.write_bytes(uploaded[csv_names[0]])
        del uploaded
print(EXCLUSIONS.read_text(encoding='utf-8-sig')[:5000])
''')
    cell('markdown', '## 5. Build the six-class dataset once\n\nImage bytes stay unchanged; bottle rows are removed. Mixed images and bottle-only background negatives are retained. This creates one group-aware split used by every candidate. Previously held-out photos remain outside training. Reopening the notebook verifies the saved dataset instead of rebuilding a different split.\n')
    cell('code', '''if DATA.exists():
    cli('verify', '--dataset', DATA)
else:
    cli('prepare', '--archive', ARCHIVE, '--exclusions', EXCLUSIONS, '--output', DATA)
''')
    cell('markdown', '''## 6. Train the baseline — 40 epochs

This starts real GPU training. It retrains the six-class model from general pretrained YOLO11s weights. It does not resume the old seven-class checkpoint. Use the epoch timings to estimate the remaining campaign. Standard best/last checkpoints are saved after epochs. If an interrupted run has a verified failure receipt, the campaign can resume it; an abrupt runtime kill may need manual recovery.
''')
    cell('code', '''# Run the first stage only. The campaign cell below skips this run once verified complete.
baseline = RUNS / 'baseline_seed57'
if (baseline / 'run.json').is_file():
    baseline_record = json.loads((baseline / 'run.json').read_text())
    if baseline_record.get('status') == 'complete':
        report = json.loads(subprocess.check_output([
            sys.executable, str(CLI), 'status', '--dataset', str(DATA), '--runs', str(RUNS)], text=True))
        assert {'trial': 'baseline', 'seed': 57} in report['completed_runs'], 'Baseline provenance is not valid.'
        print(json.dumps(report, indent=2))
    else:
        cli('train', '--dataset', DATA, '--output', baseline, '--trial', 'baseline', '--device', '0', '--resume')
else:
    cli('train', '--dataset', DATA, '--output', baseline, '--trial', 'baseline', '--device', '0')
''')
    cell('markdown', '''## 7. Export results now or after more trials

Run this cell after the baseline and again whenever you want to return partial results. The ZIP includes dataset provenance and run reports, logs, curves, per-image matching stats and overlays. Model weights stay in the persistent run folders so the report remains manageable. No accuracy claim is made if a run failed or remains incomplete.
''')
    cell('code', '''EXPORT = Path('/content/V5_Six_Class_Unified_Results.zip')
with zipfile.ZipFile(EXPORT, 'w', zipfile.ZIP_DEFLATED) as z:
    for name in ('dataset_receipt.json', 'data.yaml', 'train.txt', 'val.txt', 'test.txt'):
        p = DATA / name
        if p.is_file(): z.write(p, 'data/' + name)
    if RUNS.exists():
        for p in sorted(RUNS.rglob('*')):
            if p.is_file() and p.suffix.lower() in {'.json', '.csv', '.yaml', '.txt', '.png', '.jpg'}:
                z.write(p, 'runs/' + str(p.relative_to(RUNS)))
print('Report ZIP:', EXPORT, 'bytes:', EXPORT.stat().st_size)
files.download(str(EXPORT))
''')
    cell('markdown', '''## 8. Run the six unified variants and repeat confirmations

Run this only when ready for the remaining GPU time. It runs U1 (gentler augmentation), U2 (1024 pixels), U3 (bounded weak-class balancing), U4 (lower learning rate), U5 (YOLO11m), and U6 (supported combination or 80-epoch convergence). Trials normally use 40 epochs. It then repeats the baseline and selected candidate at seeds 58 and 59. Valid completed runs are skipped, allowing the campaign to continue across sessions.

Afterward rerun the export cell. Return the ZIP for analysis of class support, repeat variability and paired group-bootstrap intervals. Specialist training is deliberately unavailable here and will follow only if the measured results justify it. Reserved test images are never used to pick a candidate.
''')
    cell('code', '''cli('campaign', '--dataset', DATA, '--runs', RUNS, '--device', '0')
cli('status', '--dataset', DATA, '--runs', RUNS)
''')
    notebook = dict(cells=cells, metadata=dict(
        kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python', version='3.12'), accelerator='GPU'),
        nbformat=4, nbformat_minor=5)
    for i, item in enumerate(cells):
        item['id'] = f'v5-six-{i:02d}'
    output = Path(output)
    output.write_text(json.dumps(notebook, indent=1) + '\n')
    return output


if __name__ == '__main__':
    print(build())
