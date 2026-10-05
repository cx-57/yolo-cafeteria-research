import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("labels_dir")
parser.add_argument("--num-classes", type=int, required=True)
args = parser.parse_args()

labels_dir = Path(args.labels_dir)
errors = []

for label_file in labels_dir.rglob("*.txt"):
    for line_number, line in enumerate(label_file.read_text().splitlines(), start=1):
        parts = line.split()

        if len(parts) != 5:
            errors.append(
                f"{label_file}:{line_number}: expected 5 values"
            )
            continue

        try:
            class_id = int(parts[0])
            x, y, width, height = map(float, parts[1:])
        except ValueError:
            errors.append(
                f"{label_file}:{line_number}: invalid number"
            )
            continue

        if class_id < 0 or class_id >= args.num_classes:
            errors.append(
                f"{label_file}:{line_number}: class {class_id} is invalid"
            )

        for value in [x, y, width, height]:
            if value < 0 or value > 1:
                errors.append(
                    f"{label_file}:{line_number}: coordinate outside 0-1"
                )
                break

        if width <= 0 or height <= 0:
            errors.append(
                f"{label_file}:{line_number}: box size must be positive"
            )

if errors:
    for error in errors:
        print(error)
else:
    print("All labels valid.")
