"""Validate normalized YOLO bounding-box labels."""

from __future__ import annotations
import argparse
from pathlib import Path

def validate_file(path: Path, num_classes: int) -> list[str]:
    errors = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path}:{line_no}: expected 5 fields")
            continue
        try:
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            errors.append(f"{path}:{line_no}: non-numeric label")
            continue
        if not 0 <= cls < num_classes:
            errors.append(f"{path}:{line_no}: class {cls} out of range")
        if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
            errors.append(f"{path}:{line_no}: coordinates outside [0,1]")
        if w <= 0 or h <= 0:
            errors.append(f"{path}:{line_no}: non-positive box size")
    return errors

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels_dir", type=Path)
    parser.add_argument("--num-classes", type=int, required=True)
    args = parser.parse_args()
    errors = []
    for path in args.labels_dir.rglob("*.txt"):
        errors.extend(validate_file(path, args.num_classes))
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print("All labels valid.")

if __name__ == "__main__":
    main()
