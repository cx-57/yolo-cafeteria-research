"""Count images and object instances for selected YOLO class IDs.

Use after exporting a candidate dataset locally. This reports source-class counts;
it does NOT decide whether an image is a strict NHS visual match.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path


def audit(labels_dir: Path, class_ids: set[int]) -> dict[int, dict[str, int]]:
    images_by_class: dict[int, set[str]] = defaultdict(set)
    instances: Counter[int] = Counter()

    for label_path in labels_dir.rglob("*.txt"):
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(parts[0])
            except ValueError:
                continue
            if class_id not in class_ids:
                continue
            images_by_class[class_id].add(label_path.stem)
            instances[class_id] += 1

    return {
        class_id: {
            "images": len(images_by_class[class_id]),
            "instances": instances[class_id],
        }
        for class_id in sorted(class_ids)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels_dir", type=Path)
    parser.add_argument(
        "--class-ids",
        required=True,
        help="Comma-separated source class IDs, e.g. 2,7,14",
    )
    args = parser.parse_args()

    ids = {int(value) for value in args.class_ids.split(",")}
    results = audit(args.labels_dir, ids)
    for class_id, counts in results.items():
        print(f"class {class_id}: {counts['images']} images, {counts['instances']} instances")


if __name__ == "__main__":
    main()
