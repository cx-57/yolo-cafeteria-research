#!/usr/bin/env bash
# Publish ONLY the verified 704-image grouped dataset from a local experiment ZIP.
# Usage (inside experiments/yolo-v2-704):
#   bash publish_dataset.sh /path/to/YOLO_V2_704_GroupSplit_Trial.zip
set -euo pipefail

cd "$(dirname "$0")"
EXPECTED_NAME="YOLO_V2_704_GroupSplit_Trial.zip"
EXPECTED_SHA="16c9990b2cf45a14915ae3344307f4c1bf8dc9dbbd91a31d0b697313a8c21e27"
ZIP_PATH="${1:-$EXPECTED_NAME}"

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "Missing combined ZIP: $ZIP_PATH"
  echo "Download $EXPECTED_NAME from the previous YOLO chat and pass its path."
  exit 1
fi

if ! command -v git >/dev/null || ! command -v unzip >/dev/null || ! command -v shasum >/dev/null; then
  echo "Required tools: git, unzip, shasum (included on macOS)."
  exit 1
fi

if [[ "$(git rev-parse --is-inside-work-tree 2>/dev/null)" != "true" ]]; then
  echo "Run inside the cloned GitHub repository."
  exit 1
fi
BRANCH="$(git branch --show-current)"
if [[ -z "$BRANCH" ]]; then
  echo "Checkout a local branch before publishing; detached HEAD is not supported."
  exit 1
fi
if ! git diff --cached --quiet; then
  echo "You have other staged changes. Commit or unstage those first."
  exit 1
fi

ACTUAL_SHA="$(shasum -a 256 "$ZIP_PATH" | awk '{print $1}')"
if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
  echo "SHA-256 mismatch: this is NOT the verified 704-image combined ZIP."
  echo "Expected: $EXPECTED_SHA"
  echo "Actual:   $ACTUAL_SHA"
  exit 1
fi

echo "ZIP hash verified. Extracting..."
unzip -oq "$ZIP_PATH" -d .
if [[ ! -f "yolo_v2_trial_704/data.yaml" || ! -f "yolo_v2_trial_704/source_manifest.json" ]]; then
  echo "Combined dataset lacks data.yaml or source_manifest.json."
  exit 1
fi

echo "Checking all images, YOLO labels, hashes, and group-safe splits..."
python3 train_yolo_v2_trial.py --verify-only

echo "Checking individual files are below GitHub's 100 MB limit..."
python3 - <<'PY'
from pathlib import Path
root = Path("yolo_v2_trial_704")
images = list((root / "images").glob("*/*"))
labels = list((root / "labels").glob("*/*.txt"))
if len(images) != 704 or len(labels) != 704:
    raise SystemExit(f"Unexpected file count: {len(images)} images and {len(labels)} labels")
files = images + labels + [root / "data.yaml", root / "source_manifest.json"]
large = [(str(f), f.stat().st_size) for f in files if f.stat().st_size >= 100_000_000]
if large:
    raise SystemExit(f"GitHub blocks files >=100 MB: {large}")
print(f"Ready: {len(images)} images, {len(labels)} labels, manifest and YAML")
PY

# Keep the 141 MB combined ZIP ignored. Track original dataset files individually.
git add -f -- yolo_v2_trial_704/images yolo_v2_trial_704/labels \
  yolo_v2_trial_704/source_manifest.json yolo_v2_trial_704/data.yaml

if git diff --cached --quiet; then
  echo "All dataset files are already committed. Nothing to upload."
  exit 0
fi
echo "Committing verified 704-image dataset on branch: $BRANCH"
git commit -m "Add verified 704-image grouped YOLO V2 experiment dataset"
git push origin "$BRANCH"
echo "Uploaded! Browse experiments/yolo-v2-704/yolo_v2_trial_704/ on GitHub."
