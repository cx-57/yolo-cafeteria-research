import argparse
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root")
args = parser.parse_args()

root = Path(args.root)
image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

images = []
for file in root.rglob("*"):
    if file.suffix.lower() in image_extensions:
        images.append(file)

label_files = list(root.rglob("*.txt"))
class_counts = Counter()
bad_rows = 0

for label_file in label_files:
    for line in label_file.read_text().splitlines():
        parts = line.split()

        if len(parts) < 5:
            bad_rows += 1
            continue

        try:
            class_id = int(parts[0])
            class_counts[class_id] += 1
        except ValueError:
            bad_rows += 1

print("Images:", len(images))
print("Label files:", len(label_files))
print("Instances by class:", dict(sorted(class_counts.items())))
print("Bad label rows:", bad_rows)
