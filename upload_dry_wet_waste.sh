#!/usr/bin/env bash
set -euo pipefail

# Upload the complete screened 233-image Dry & Wet subset (not the unreviewed 1,826).
cd "$(git rev-parse --show-toplevel)"
[[ "$(git remote get-url origin)" == *"cx-57/yolo-cafeteria-research"* ]] || {
  echo "Wrong GitHub repository" >&2; exit 1;
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Commit or stash tracked changes first" >&2; exit 1;
}
git switch main
git pull --ff-only origin main

ZIP="${1:-$HOME/Downloads/Dry_Wet_Waste_NHS_233.zip}"
[[ -f "$ZIP" ]] || {
  echo "Missing file: $ZIP" >&2
  echo "Download Dry_Wet_Waste_NHS_233.zip to Downloads, or pass its path." >&2
  exit 1
}
python3 - "$ZIP" <<'PY'
import csv
import hashlib
import io
import math
import sys
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

target = Path("data/processed/dry_wet_waste")
manifest_path = Path("data/manifests/dry_wet_waste.csv")
with ZipFile(Path(sys.argv[1]).expanduser()) as z:
    assert z.testzip() is None, "Corrupted ZIP"
    images = {Path(n).stem: n for n in z.namelist() if n.startswith("images/") and n.lower().endswith((".jpg", ".jpeg", ".png"))}
    labels = {Path(n).stem: n for n in z.namelist() if n.startswith("labels/") and n.endswith(".txt")}
    assert len(images) == len(labels) == 233 and images.keys() == labels.keys(), "Expected 233 image/label pairs"
    manifest = z.read("manifest.csv")
    rows = list(csv.DictReader(io.StringIO(manifest.decode("utf-8-sig"))))
    assert len(rows) == 233, "Expected 233 manifest rows"
    assert {r["processed_image"] for r in rows} == {Path(n).name for n in images.values()}, "Manifest mismatch"
    counts = Counter()
    for row in rows:
        img = z.read("images/" + row["processed_image"])
        assert hashlib.sha256(img).hexdigest() == row["sha256"], "Image checksum mismatch"
    for name in labels.values():
        parts = z.read(name).decode().split()
        assert len(parts) == 5, f"Invalid bounding box: {name}"
        cid = int(parts[0])
        x, y, w, h = map(float, parts[1:])
        assert cid in (5, 8, 9) and all(map(math.isfinite, (x, y, w, h)))
        assert w > 0 and h > 0 and x-w/2 >= -1e-6 and x+w/2 <= 1+1e-6
        assert y-h/2 >= -1e-6 and y+h/2 <= 1+1e-6
        counts[cid] += 1
    assert counts == {5: 34, 8: 56, 9: 143}, f"Wrong class totals: {counts}"
    if manifest_path.exists():
        assert manifest_path.read_bytes() == manifest, "Existing manifest conflicts"
    for selected in (images, labels):
        folder = target / ("images" if selected is images else "labels")
        if folder.is_dir():
            extras = {p.name for p in folder.iterdir() if p.is_file()} - {Path(n).name for n in selected.values()}
            assert not extras, f"Unexpected existing files: {sorted(extras)[:3]}"
        for name in selected.values():
            file = target / name
            assert not file.exists() or file.read_bytes() == z.read(name), f"Conflict: {file}"
    for selected in (images, labels):
        for name in selected.values():
            file = target / name
            file.parent.mkdir(parents=True, exist_ok=True)
            if not file.exists():
                file.write_bytes(z.read(name))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        manifest_path.write_bytes(manifest)
print("Verified: 233 images, 233 labels: 143 bottles, 56 wrappers, 34 napkins.")
print("TRAINING ONLY — use independent NHS photos for evaluation.")
PY

git add data/processed/dry_wet_waste data/manifests/dry_wet_waste.csv
if ! git diff --cached --quiet; then
  git commit -m "Add 233 screened Dry and Wet Waste training photos"
fi
git push origin main
echo "Upload complete: data/processed/dry_wet_waste/"
