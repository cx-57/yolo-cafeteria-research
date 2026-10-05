"""Remap YOLO class IDs using an explicit old->new mapping."""

from __future__ import annotations
import argparse
from pathlib import Path

def remap_label_file(path: Path, mapping: dict[int, int], drop_unmapped: bool = True) -> None:
    out: list[str] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        old_id = int(parts[0])
        if old_id not in mapping:
            if drop_unmapped:
                continue
            out.append(line)
            continue
        parts[0] = str(mapping[old_id])
        out.append(" ".join(parts))
    path.write_text("\n".join(out) + ("\n" if out else ""))

def parse_mapping(text: str) -> dict[int, int]:
    result = {}
    for pair in text.split(","):
        old, new = pair.split(":")
        result[int(old)] = int(new)
    return result

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels_dir", type=Path)
    parser.add_argument("--mapping", required=True, help='Example: "0:0,3:0,5:2"')
    parser.add_argument("--keep-unmapped", action="store_true")
    args = parser.parse_args()
    mapping = parse_mapping(args.mapping)
    for path in args.labels_dir.rglob("*.txt"):
        remap_label_file(path, mapping, not args.keep_unmapped)

if __name__ == "__main__":
    main()
