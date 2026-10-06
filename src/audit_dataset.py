import argparse
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root")
args = parser.parse_args()

root = Path(args.root)
image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

images = [
    file for file in root.rglob("*")
    if file.suffix.lower() in image_extensions
]

labels = list(root.rglob("*.txt"))
class_counts = Counter()
bad_rows = 0

for label in labels:
    for line in label.read_text().splitlines():
        parts = line.split()

        if len(parts) != 5:
            bad_rows += 1
            continue

        try:
            class_id = int(parts[0])
            x, y, width, height = map(float, parts[1:])
        except ValueError:
            bad_rows += 1
            continue

        if not all(0 <= value <= 1 for value in [x, y, width, height]):
            bad_rows += 1
            continue

        if width <= 0 or height <= 0:
            bad_rows += 1
            continue

        class_counts[class_id] += 1

print("Images:", len(images))
print("Label files:", len(labels))
print("Instances by class:", dict(sorted(class_counts.items())))
print("Bad label rows:", bad_rows)
