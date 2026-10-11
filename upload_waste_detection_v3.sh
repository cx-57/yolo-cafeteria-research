#!/usr/bin/env bash
set -euo pipefail

# Upload the audited, unthinned 331-photo NHS subset (not the old 51-photo ZIP).
cd "$(git rev-parse --show-toplevel)"
[[ "$(git remote get-url origin)" == *"cx-57/yolo-cafeteria-research"* ]] || {
  echo "Wrong GitHub repository" >&2; exit 1;
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Commit or stash local tracked changes first" >&2; exit 1;
}
git switch main
git pull --ff-only origin main

ZIP="${1:-$HOME/Downloads/Waste_Detection_v3_NHS_Untrimmed_331.zip}"
python3 - "$ZIP" <<'PY'
import csv
import io
import math
import sys
from pathlib import Path
from zipfile import ZipFile

archive = Path(sys.argv[1]).expanduser()
if not archive.is_file():
    sys.exit(f"Missing dataset ZIP: {archive}\nDownload it from your ChatGPT conversation first.")
target = Path("data/processed/waste_detection_v3")
manifest_target = Path("data/manifests/waste_detection_v3.csv")
with ZipFile(archive) as z:
    names = z.namelist()
    if z.testzip() is not None:
        sys.exit("ZIP failed its checksum test")
    images = {Path(n).stem: n for n in names if n.startswith("images/") and n.lower().endswith((".jpg", ".jpeg", ".png"))}
    labels = {Path(n).stem: n for n in names if n.startswith("labels/") and n.endswith(".txt")}
    if len(images) != 331 or len(labels) != 331 or images.keys() != labels.keys():
        sys.exit(f"Expected exactly 331 matching pairs; got {len(images)} images, {len(labels)} labels")
    if "manifest.csv" not in names:
        sys.exit("Missing source-provenance manifest.csv")
    rows = list(csv.DictReader(io.StringIO(z.read("manifest.csv").decode("utf-8-sig"))))
    if len(rows) != 331:
        sys.exit(f"Expected 331 provenance rows, got {len(rows)}")
    box_count, bowl_count = 0, 0
    for stem in sorted(images):
        img = z.read(images[stem])
        if not img.startswith(b"\\xff\\xd8") and not img.startswith(b"\\x89PNG"):
            sys.exit(f"Unsupported or damaged image: {images[stem]}")
        lines = z.read(labels[stem]).decode("utf-8").splitlines()
        if not lines:
            sys.exit(f"Missing labels for {images[stem]}")
        for line in lines:
            p = line.split()
            if len(p) != 5:
                sys.exit(f"Not a YOLO detection box: {labels[stem]}")
            try:
                cid = int(p[0])
                x, y, w, h = map(float, p[1:])
            except ValueError:
                sys.exit(f"Bad annotation: {labels[stem]}")
            if not (0 <= cid <= 10 and all(map(math.isfinite, (x,y,w,h)))
                    and w > 0 and h > 0 and x-w/2 >= -1e-6
                    and y-h/2 >= -1e-6 and x+w/2 <= 1+1e-6
                    and y+h/2 <= 1+1e-6):
                sys.exit(f"Out-of-bounds class/box: {labels[stem]}")
            box_count += 1
            bowl_count += cid == 10
    if box_count != 342 or bowl_count != 82:
        sys.exit(f"Unexpected annotations: {box_count} boxes, {bowl_count} bowl boxes")
    for folder, selected in (("images", images), ("labels", labels)):
        dest = target / folder
        if dest.is_dir():
            extras = {p.name for p in dest.iterdir() if p.is_file()} - {Path(n).name for n in selected.values()}
            if extras:
                sys.exit(f"{dest} contains unexpected old files: {sorted(extras)[:3]}")
    for selected in (images, labels):
        for n in selected.values():
            destination = target / n
            payload = z.read(n)
            if destination.exists() and destination.read_bytes() != payload:
                sys.exit(f"Refusing to replace conflicting existing file: {destination}")
    if manifest_target.exists() and manifest_target.read_bytes() != z.read("manifest.csv"):
        sys.exit(f"Refusing to overwrite a different manifest: {manifest_target}")
    for selected in (images, labels):
        for n in selected.values():
            dest = target / n
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.write_bytes(z.read(n))
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    if not manifest_target.exists():
        manifest_target.write_bytes(z.read("manifest.csv"))

print("Verified and extracted: 331 image/label pairs, 342 boxes (82 bowl boxes).")
print("All uploaded images are training-only; no source validation/test split is trusted.")
PY

git add data/processed/waste_detection_v3 data/manifests/waste_detection_v3.csv
if git diff --cached --quiet; then
  echo "All 331 images are already committed locally."
else
  git commit -m "Add 331 audited Waste Detection v3 training photos"
fi
git push origin main
echo "Finished. Verify at GitHub: data/processed/waste_detection_v3/"
