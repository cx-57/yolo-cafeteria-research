"""Local data and runtime paths, kept outside the source repository."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORE = Path(os.environ.get('CAFETERIA_STORAGE', ROOT.parent / (ROOT.name + '-assets')))
DATA = STORE / 'data'
ARTIFACTS = STORE / 'artifacts'
RUNTIME = Path(os.environ.get('CAFETERIA_RUNTIME', Path.home() / 'Library/Caches/yolo-cafeteria-runtime'))
EVALUATION = ARTIFACTS / 'simple-specialists/shared-eight-reviewed.json'


def resolve_image(path, root=ROOT, data=DATA):
    path = Path(path).expanduser()
    if path.is_file():
        return path
    try:
        candidate = data / path.relative_to(root / 'data')
    except ValueError:
        candidate = None
    if candidate is not None and candidate.is_file():
        return candidate
    # Historical manifests can point to a different user's runtime cache.
    if 'yolo-cafeteria-runtime' in path.parts:
        candidate = RUNTIME.joinpath(*path.parts[path.parts.index('yolo-cafeteria-runtime') + 1:])
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'Image unavailable: {path}. Set CAFETERIA_STORAGE/CAFETERIA_RUNTIME or update the manifest.')
