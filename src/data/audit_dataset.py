"""Audit a YOLO-format dataset."""

from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

def audit_dataset(root: Path) -> dict:
    images = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES]
    labels = list(root.rglob("*.txt"))
    class_counts: Counter[int] = Counter()
    malformed = 0
    for label in labels:
        for line in label.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 5:
                malformed += 1
                continue
            try:
                class_counts[int(parts[0])] += 1
            except ValueError:
                malformed += 1
    return {
        "images": len(images),
        "label_files": len(labels),
        "instances_by_class": dict(sorted(class_counts.items())),
        "malformed_rows": malformed,
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(audit_dataset(args.root))

if __name__ == "__main__":
    main()
