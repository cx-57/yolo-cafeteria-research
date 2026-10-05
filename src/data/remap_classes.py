import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("labels_dir")
parser.add_argument("--mapping", required=True, help='Example: "0:0,3:0,5:2"')
parser.add_argument("--keep-unmapped", action="store_true")
args = parser.parse_args()

labels_dir = Path(args.labels_dir)

mapping = {}
for pair in args.mapping.split(","):
    old_class, new_class = pair.split(":")
    mapping[int(old_class)] = int(new_class)

for label_file in labels_dir.rglob("*.txt"):
    new_lines = []

    for line in label_file.read_text().splitlines():
        parts = line.split()

        if len(parts) < 5:
            continue

        old_class = int(parts[0])

        if old_class in mapping:
            parts[0] = str(mapping[old_class])
            new_lines.append(" ".join(parts))

        elif args.keep_unmapped:
            new_lines.append(line)

    label_file.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))

print("Done remapping classes.")
