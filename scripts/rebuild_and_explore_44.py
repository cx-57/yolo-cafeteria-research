"""Recreate the exact screened 44 public images on this Mac, then explore.

Downloads each original from its public source, SHA-verifies it against the
November 1 screening cohort, never claims independent labels or accuracy.
"""
import argparse
import csv
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import explore_candidates as exploratory
import stage_commons_candidates as commons
import stage_trashbox_candidates as trashbox

KEEP = {
    'commons': set('''9cdebbf715fd67 c8c73853613120 0d4bd5502837ed b9303ac19bc1fa
80af3befea63af 3267e7d0f1f186 7d04361ebae488 d70691cf6fb611
170aabc783e264 8ab005830b3e4b 51dce66fad1a6b e31dc6bddb560e
1fb76d87e3b824 fc333b6cd7edfb 825c26373b6cc1 455e36ac50f82a
1150b10a8dca70 532e1fc9a8f416 3e62425f644c07 a29479e8818136
a39bde5e225af8 03818f612fce2f c477fd37f0a040 8f479d4700d36e
b4a62d54de76b7 120667189b848c ac2dc4bd91b1d1 abb8df93cde3e0
de23023b4a3057'''.split()),
    'trashbox': set('''1c752f7f68a1d3 0e99f045921b2a 5b1e9deb425dbf
cbb56c5a0852ab c8523bf8835555 5624478ce42042 f9e2507b47dca9
651b8babe42679 76204f9caff8e3 825e1bd1ffb7c1 8fcb86c4fde0e0
92b4fb8feca052 3d82a2aa685cc1 c2d246e7a4c884 c5969032c3cf14'''.split())
}
EXPECTED = {'commons': 29, 'trashbox': 15}


def select(staging, group):
    """Pick exact reviewed byte identities; mismatch means stop, never substitute."""
    with (staging / 'review.csv').open(newline='', encoding='utf-8') as f:
        records = list(csv.DictReader(f))
    accepted = {}
    for row in records:
        sha = row['sha256']
        if any(sha.startswith(prefix) for prefix in KEEP[group]):
            path = staging / 'images' / row['filename']
            if not path.is_file():
                raise FileNotFoundError(path)
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != sha:
                raise ValueError('Downloaded source image was altered: ' + row['filename'])
            if sha in accepted:
                raise ValueError('Duplicate download image SHA')
            accepted[sha] = (row, data)
    present = {sha[:14] for sha in accepted}
    if present != KEEP[group] or len(accepted) != EXPECTED[group]:
        missing = sorted(KEEP[group] - present)
        raise ValueError(f'{group}: expected {EXPECTED[group]} exact candidates, received '
                         f'{len(accepted)}. Missing SHA prefixes: {missing}. '
                         'Source collection has changed or the network blocked downloads; '
                         'not substituting different pictures.')
    return accepted


def build_zip(trashbox_staging, commons_staging, destination):
    """Produce compatible unverified manifest ZIP without model predictions."""
    if destination.exists():
        raise FileExistsError(destination)
    combined = []
    for group, staging in [('commons', commons_staging), ('trashbox', trashbox_staging)]:
        for sha, (row, data) in sorted(select(staging, group).items()):
            name = f'images/{group}__{row["filename"]}'
            source = ('https://commons.wikimedia.org' if group == 'commons'
                      else 'https://github.com/nikhilvenkatkumsetty/TrashBox')
            combined.append((dict(
                path=name, sha256=sha, source=source, source_hint=(
                    row['project_class_hint'] if group == 'commons' else row['source_hint']),
                original_path=(row['commons_page'] if group == 'commons'
                               else row['original_path']),
                license=(row['license'] if group == 'commons' else 'NOT VERIFIED'),
                creator=(row.get('creator_html', '') if group == 'commons' else ''),
                license_url=row.get('license_url', '') if group == 'commons' else '',
                independent=False, human_reviewed=False,
                scoring_eligible=(row['filename'] not in {
                    'bottle_09_c8523bf883.jpg', 'bottle_10_5624478ce4.jpg',
                    'bottle_13_f9e2507b47.jpg'}),
                draft_boxes=[]), data))
    manifest = dict(schema_version=2,
        status='RECONSTRUCTED_PUBLIC_SOURCES_FOR_EXPLORATORY_INFERENCE',
        independent_certified_images=0, human_verified_images=0,
        verified_bin_labels=0, images=[metadata for metadata,_ in combined])
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_STORED) as z:
        z.writestr('manifest.json', json.dumps(manifest, indent=2))
        for metadata,data in combined:
            z.writestr(metadata['path'], data)
    return manifest


def run(out, device):
    if out.exists():
        raise FileExistsError('Refusing to overwrite existing output: '+str(out))
    # Avoid network costs if model is absent/changed.
    exploratory.check_frozen(exploratory.s.ROOT)
    with tempfile.TemporaryDirectory(prefix='cafeteria-44-fetch-') as td:
        root = Path(td)
        print('Downloading the exact original screened public source photos...', flush=True)
        trashbox.stage(root / 'trashbox', quota=16, seed=20261101)
        commons.build_review(root / 'commons', target_each=16, seed=20261101)
        bundle = root / 'reviewed-44-source-bundle.zip'
        manifest = build_zip(root / 'trashbox', root / 'commons', bundle)
        print(f"Verified {len(manifest['images'])} original candidate bytes; running frozen models.", flush=True)
        return exploratory.run(bundle, out, device)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, type=Path,
                   help='New, nonexistent output directory on the Mac')
    p.add_argument('--device', default='mps', choices=['mps', 'cpu'])
    args = p.parse_args()
    result = run(args.out.expanduser().resolve(), args.device)
    print(json.dumps({k:v for k,v in result.items() if k != 'per_image'},indent=2))
