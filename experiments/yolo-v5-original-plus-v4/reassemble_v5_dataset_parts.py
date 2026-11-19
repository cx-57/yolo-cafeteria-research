"""Reassemble four byte-identical parts of the V5 audit320 dataset ZIP.

Put the four files in the same folder, and run:
    python reassemble_v5_dataset_parts.py /path/to/folder
The output is YOLO_V5_partial_annotation_audit320.zip and its SHA256 is verified.
"""
from pathlib import Path
import hashlib
import sys

folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
filename = "YOLO_V5_partial_annotation_audit320.zip"
expected = "4410e575f3797acca490c240fd58b85113aefb1261157c24c257b5dbeb2ecd9b"
parts = [folder / f"{filename}.part{i:02d}" for i in range(1, 5)]
missing = [p.name for p in parts if not p.is_file()]
if missing:
    raise FileNotFoundError(f"Missing ZIP segments: {missing}")

output = folder / filename
digest = hashlib.sha256()
with output.open("wb") as dst:
    for part in parts:
        with part.open("rb") as src:
            while True:
                block = src.read(8 * 1024 * 1024)
                if not block:
                    break
                dst.write(block)
                digest.update(block)
actual = digest.hexdigest()
if actual != expected:
    output.unlink()
    raise RuntimeError(f"Checksum mismatch: expected {expected}; found {actual}")
print(f"Reassembled and SHA256-verified: {output}")
