#!/usr/bin/env bash
set -euo pipefail

# Run from the YOLO repository after saving both ZIPs in ~/Downloads.
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

if [[ "$(git remote get-url origin)" != *"cx-57/yolo-cafeteria-research"* ]]; then
    echo "Wrong repository. Open yolo-cafeteria-research in Terminal." >&2
    exit 1
fi

RAW="$HOME/Downloads/YOLO_Raw_Images.zip"
PROCESSED="$HOME/Downloads/YOLO_Processed_Images.zip"

for archive in "$RAW" "$PROCESSED"; do
    if [[ ! -f "$archive" ]]; then
        echo "Missing $archive. Download both dataset ZIPs into ~/Downloads." >&2
        exit 1
    fi
done

# Don't mix this dataset commit with unfinished code edits.
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Commit or stash your existing tracked changes first." >&2
    exit 1
fi

git switch main
git pull --ff-only origin main

# The archives contain paths starting with data/, so extract at repo root.
unzip -tqq "$RAW"
unzip -tqq "$PROCESSED"
unzip -oq "$RAW" -d "$ROOT"
unzip -oq "$PROCESSED" -d "$ROOT"

# Confirm every image from both archives exists, with a matching YOLO label.
python3 - "$RAW" "$PROCESSED" <<'PY'
import sys
from pathlib import Path
from zipfile import ZipFile

expected = {"raw": 2004, "processed": 1960, "candidates": 44}
counts = {part: 0 for part in expected}
suffixes = (".jpg", ".jpeg", ".png", ".webp")

for archive in sys.argv[1:]:
    with ZipFile(archive) as source:
        for filename in source.namelist():
            path = Path(filename)
            if path.suffix.lower() not in suffixes or path.parent.name != "images":
                continue
            part = path.parts[1]
            if part not in counts:
                continue
            counts[part] += 1
            if not path.is_file():
                sys.exit(f"Missing extracted image: {path}")
            if part in ("processed", "candidates"):
                label = path.parent.parent / "labels" / (path.stem + ".txt")
                if not label.is_file():
                    sys.exit(f"Missing YOLO label: {label}")

for part, target in expected.items():
    print(f"{part}: {counts[part]} images (expected {target})")
    if counts[part] != target:
        sys.exit(f"Wrong count for {part}; upload cancelled.")
print("All packaged photos and YOLO labels verified.")
PY

git add -A data/
if git diff --cached --quiet; then
    echo "All dataset images are already committed."
else
    git commit -m "Add raw and processed YOLO images including LVIS plates"
fi

git push origin main
echo "Done. Your image files are now pushed to GitHub."
