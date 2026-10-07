#!/usr/bin/env bash
set -euo pipefail

ZIP_PATH="${1:-$HOME/Downloads/nhs_reviewed_batch_repo_ready.zip}"

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "Could not find: $ZIP_PATH"
  echo "Download nhs_reviewed_batch_repo_ready.zip, then run:"
  echo "  bash scripts/import_reviewed_batch.sh /path/to/nhs_reviewed_batch_repo_ready.zip"
  exit 1
fi

git pull --ff-only
unzip -oq "$ZIP_PATH" -d .

python3 - <<'PY'
from pathlib import Path

expected = {
    "school_lunch": (467, {0, 7}),
    "tetra_pak": (661, {7}),
    "macworx": (121, {5}),
    "utensil": (128, {3}),
}

for source, (expected_count, allowed_classes) in expected.items():
    root = Path("data/processed") / source
    images = sorted(p for p in (root / "images").iterdir() if p.is_file())
    labels = sorted((root / "labels").glob("*.txt"))

    assert len(images) == expected_count, (
        f"{source}: expected {expected_count} images, found {len(images)}"
    )
    assert len(labels) == expected_count, (
        f"{source}: expected {expected_count} labels, found {len(labels)}"
    )

    for image in images:
        label = root / "labels" / f"{image.stem}.txt"
        assert label.exists(), f"{source}: missing label for {image.name}"

    instances = 0
    seen_classes = set()
    for label in labels:
        for line in label.read_text().splitlines():
            parts = line.split()
            assert len(parts) == 5, f"{label}: non-detection row: {line}"
            class_id = int(parts[0])
            values = [float(v) for v in parts[1:]]
            assert class_id in allowed_classes, (
                f"{label}: unexpected class {class_id}; allowed {sorted(allowed_classes)}"
            )
            assert all(0 <= v <= 1 for v in values), f"{label}: out-of-range box"
            assert values[2] > 0 and values[3] > 0, f"{label}: invalid box size"
            seen_classes.add(class_id)
            instances += 1

    print(
        f"{source}: validated {len(images)} images, "
        f"{instances} instances, classes {sorted(seen_classes)}"
    )

print("All manually reviewed subsets validated.")
PY

git add   data/processed/school_lunch   data/processed/tetra_pak   data/processed/macworx   data/processed/utensil   data/manifests   data/audits

if git diff --cached --quiet; then
  echo "No new reviewed dataset files to commit."
else
  git commit -m "Add manually reviewed NHS dataset subsets"
  git push
fi

echo "Done."
