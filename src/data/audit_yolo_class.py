import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("labels_dir")
parser.add_argument("--class-ids", required=True, help="Example: 2,7,14")
args = parser.parse_args()

labels_dir = Path(args.labels_dir)
class_ids = [int(x) for x in args.class_ids.split(",")]

image_sets = {}
instance_counts = {}

for class_id in class_ids:
    image_sets[class_id] = set()
    instance_counts[class_id] = 0

for label_file in labels_dir.rglob("*.txt"):
    for line in label_file.read_text().splitlines():
        parts = line.split()

        if len(parts) < 5:
            continue

        try:
            class_id = int(parts[0])
        except ValueError:
            continue

        if class_id in class_ids:
            image_sets[class_id].add(label_file.stem)
            instance_counts[class_id] += 1

for class_id in class_ids:
    print(
        "Class",
        class_id,
        "-",
        len(image_sets[class_id]),
        "images,",
        instance_counts[class_id],
        "instances",
    )
