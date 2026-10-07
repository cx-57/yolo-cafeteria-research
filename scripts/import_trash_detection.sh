#!/usr/bin/env bash
set -euo pipefail

ZIP_PATH="${1:-$HOME/Downloads/trash_detection_nhs_strict_repo_ready.zip}"

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "Could not find: $ZIP_PATH"
  echo "Download trash_detection_nhs_strict_repo_ready.zip, then run:"
  echo "  bash scripts/import_trash_detection.sh /path/to/trash_detection_nhs_strict_repo_ready.zip"
  exit 1
fi

git pull --ff-only
unzip -oq "$ZIP_PATH" -d .

python3 - <<'PY'
from pathlib import Path
import csv

root = Path("data/processed/trash_detection")
images = sorted(p for p in (root / "images").iterdir() if p.is_file())
labels = sorted((root / "labels").glob("*.txt"))

assert len(images) == 197, f"Expected 197 images, found {len(images)}"
assert len(labels) == 197, f"Expected 197 labels, found {len(labels)}"

allowed = {9, 10}
instances = 0
by_class = {9: 0, 10: 0}

for image in images:
    label = root / "labels" / f"{image.stem}.txt"
    assert label.exists(), f"Missing label for {image.name}"

for label in labels:
    for line in label.read_text().splitlines():
        parts = line.split()
        assert len(parts) == 5, f"{label}: invalid detection row: {line}"
        cid = int(parts[0])
        values = [float(v) for v in parts[1:]]
        assert cid in allowed, f"{label}: unexpected class {cid}"
        assert all(0 <= v <= 1 for v in values), f"{label}: out-of-range box"
        assert values[2] > 0 and values[3] > 0, f"{label}: invalid box size"
        instances += 1
        by_class[cid] += 1

assert instances == 245, f"Expected 245 instances, found {instances}"
assert by_class[10] == 208, f"Expected 208 bottle instances, found {by_class[10]}"
assert by_class[9] == 37, f"Expected 37 wrapper instances, found {by_class[9]}"

audit = Path("data/audits/trash_detection_2026-10-07.csv")
manifest = Path("data/manifests/trash_detection.csv")
assert audit.exists(), "Missing per-image audit CSV"
assert manifest.exists(), "Missing retained manifest"

with audit.open(newline="") as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 2783, f"Expected 2783 audit rows, found {len(rows)}"
assert sum(r["decision"] == "KEEP" for r in rows) == 197
assert sum(r["decision"] == "SCRAP" for r in rows) == 2586

print("Trash Detection subset validated:")
print("  images:", len(images))
print("  instances:", instances)
print("  bottles:", by_class[10])
print("  wrappers:", by_class[9])
print("  audit rows:", len(rows))
PY

git add   data/processed/trash_detection   data/manifests/trash_detection.csv   data/audits/trash_detection_2026-10-07.csv   data/audits/trash_detection_summary.csv

if git diff --cached --quiet; then
  echo "No new Trash Detection dataset files to commit."
else
  git commit -m "Add reviewed Trash Detection subset"
  git push
fi

echo "Done."
