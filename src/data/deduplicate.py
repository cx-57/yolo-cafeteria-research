"""Report exact perceptual-hash duplicate groups without deleting files."""

from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import imagehash
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

def find_duplicates(root: Path) -> list[list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            with Image.open(path) as image:
                groups[str(imagehash.phash(image))].append(path)
        except Exception:
            continue
    return [paths for paths in groups.values() if len(paths) > 1]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    for group in find_duplicates(args.root):
        print("---")
        print("\n".join(map(str, group)))

if __name__ == "__main__":
    main()
