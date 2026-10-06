import argparse
from pathlib import Path

import imagehash
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("root")
args = parser.parse_args()

root = Path(args.root)
image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
hashes = {}

for file in root.rglob("*"):
    if file.suffix.lower() not in image_extensions:
        continue

    try:
        with Image.open(file) as image:
            image_hash = str(imagehash.phash(image))
    except Exception:
        continue

    hashes.setdefault(image_hash, []).append(file)

for files in hashes.values():
    if len(files) > 1:
        print("--- duplicate group ---")
        for file in files:
            print(file)
