#!/usr/bin/env bash
set -euo pipefail

ZIP_PATH="${1:-$HOME/Downloads/school_lunch_repo_ready.zip}"

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "Could not find: $ZIP_PATH"
  echo "Download school_lunch_repo_ready.zip, then run:"
  echo "  bash scripts/import_school_lunch.sh /path/to/school_lunch_repo_ready.zip"
  exit 1
fi

git pull --ff-only

unzip -oq "$ZIP_PATH" -d .

python3 - <<'PY'
from pathlib import Path

root = Path("data/processed/school_lunch")
images = sorted((root / "images").glob("*"))
labels = sorted((root / "labels").glob("*.txt"))

assert len(images) == 467, f"Expected 467 images, found {len(images)}"
assert len(labels) == 467, f"Expected 467 labels, found {len(labels)}"

bad = []
classes = set()
for label in labels:
    for line in label.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            bad.append((label, line))
            continue
        cid = int(parts[0])
        classes.add(cid)
        vals = [float(x) for x in parts[1:]]
        if cid not in {0, 7} or not all(0 <= x <= 1 for x in vals) or vals[2] <= 0 or vals[3] <= 0:
            bad.append((label, line))

assert not bad, f"Invalid label rows: {bad[:5]}"
assert classes <= {0, 7}, f"Unexpected class IDs: {classes}"

print(f"Validated {len(images)} images and {len(labels)} labels.")
print("Class IDs present:", sorted(classes))
PY

git add data/processed/school_lunch data/manifests/school_lunch.csv
git commit -m "Add reviewed school lunch subset"
git push

echo "Done."
