"""Download documented, licensed leftover-food IMAGE CANDIDATES for human review.

Does not create YOLO labels, merge data, train or evaluate any model.
Preserves receipt IDs to prevent near-identical meal views leaking between splits.
Source: https://huggingface.co/datasets/AI-ServicesBB/food-waste-dataset
License: CC BY 4.0. Source 'train' split only; test excluded.
"""
import argparse
import csv
import hashlib
import html
import io
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


SOURCE = "AI-ServicesBB/food-waste-dataset"
SOURCE_URL = "https://huggingface.co/datasets/" + SOURCE
CREDIT = "AI Service Centre Berlin-Brandenburg, Hasso-Plattner-Institut für Digital Engineering gGmbH"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run(output, limit=215):
    from datasets import load_dataset

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if limit < 1 or limit > 215:
        raise ValueError("limit must be between 1 and 215")

    print("Fetching licensed dataset train split", flush=True)
    ds = load_dataset(SOURCE, split="train")
    if len(ds) != 215:
        raise ValueError(f"Expected source train split 215, found {len(ds)}; inspect upstream changes")
    if "image" not in ds.column_names or "bonid" not in ds.column_names:
        raise ValueError("Expected image and receipt identifier; inspect upstream changes")

    root = output / "images"
    root.mkdir(parents=True)
    rows, fingerprints = [], set()
    selected = []
    by_meal = {}
    for i in range(len(ds)):
        group = str(ds[i]["bonid"])
        by_meal.setdefault(group, []).append(i)
    # Round-robin across original meal IDs for a diverse initial sample.
    for pos in range(max(map(len, by_meal.values()))):
        for group, indices in sorted(by_meal.items()):
            if pos < len(indices):
                selected.append(indices[pos])
    selected = selected[:limit]

    for n, i in enumerate(selected):
        rec = ds[i]
        group = str(rec["bonid"])
        picture = rec["image"]
        if not isinstance(picture, Image.Image):
            raise ValueError(f"Image {i} was not decoded by Hugging Face")
        image = ImageOps.exif_transpose(picture).convert("RGB")
        # Keep original resolution, no augmentation or cropping; save a JPEG copy.
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=93, subsampling=0)
        raw = buf.getvalue()
        digest = sha(raw)
        if digest in fingerprints:
            continue
        fingerprints.add(digest)
        filename = f"leftovers-{n:04d}-meal-{group}-{digest[:12]}.jpg"
        (root / filename).write_bytes(raw)
        rows.append(dict(
            filename="images/" + filename, sha256=digest,
            source_dataset=SOURCE, source_split="train", source_row=i,
            capture_group="meal-" + group, bonid=group,
            width=image.width, height=image.height,
            license="CC BY 4.0", source_url=SOURCE_URL,
            author=CREDIT, review_status="NOT_REVIEWED",
            complete_yolo_7_class_annotations="NO",
            known_target_classes="UNVERIFIED",
            independent_school_test_eligible="NO",
            historical_overlap_status="NOT_CHECKED",
            note="Test-kitchen post-meal plate, not actual school cafeteria capture"
        ))
        if (n + 1) % 40 == 0:
            print(f"saved {n+1} candidate images", flush=True)

    if not rows:
        raise ValueError("No images imported")
    with (output/"manifest.csv").open("w", newline="", encoding="utf8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(r["capture_group"] for r in rows)
    summary = dict(
        dataset=SOURCE, license="CC BY 4.0", source_url=SOURCE_URL,
        original_train=215, original_test_excluded=160,
        downloaded=len(rows), distinct_meals=len(counts), photos_per_meal=dict(counts),
        human_verified_boxes=0, yolo_training_ready=False,
        independent_school_test_eligible=False,
        caution="No seven-class boxes exist; images must be exhaustively reviewed before unified YOLO training"
    )
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (output/"README.md").write_text(
        "# CC BY 4.0 leftover-plate candidate images\n\n"
        f"Original source: {SOURCE_URL}\n\n"
        f"Copyright and credit: {CREDIT}. Licensed CC BY 4.0.\n\n"
        "These are authentic post-meal plate photos taken in a TEST KITCHEN, not images "
        "of actual school cafeteria disposal stations. Some images show different views "
        "of the same meal (see capture_group and bonid in manifest.csv). "
        "Do not split a meal across training and validation. The source's train split "
        "only was used. NO existing school test or previously exposed test images were added.\n\n"
        "**Not ready to train YOLO:** no human-reviewed complete labels for our seven classes. "
        "Do not use empty label TXT files to hide missing boxes. Verify source overlap "
        "with local project data before any merge and preserve attribution.\n\n"
        "To improve real school generalization, additionally collect actual school "
        "tray/disposal photos spanning multiple days; reserve unseen days for a sealed test.\n"
    )

    # 48-image overview with original candidate filenames/groups.
    cols, cell_w, cell_h = 6, 190, 155
    shown = rows[:48]
    sheet = Image.new("RGB", (cols*cell_w, ((len(shown)+cols-1)//cols)*cell_h+42), "white")
    painter = ImageDraw.Draw(sheet)
    painter.text((10,12), "Post-meal plate training CANDIDATES — not annotated or school-captured", fill="black")
    for i, r in enumerate(shown):
        with Image.open(output / r["filename"]) as source:
            thumb = source.copy()
        thumb.thumbnail((cell_w-10, cell_h-32))
        x, y = (i % cols)*cell_w, (i // cols)*cell_h+42
        sheet.paste(thumb, (x+(cell_w-thumb.width)//2,y))
        painter.text((x+4,y+cell_h-22),f"row {r['source_row']}  {r['capture_group']}"[:31],fill="black")
    sheet.save(output/"contact-sheet.jpg", quality=86)
    (output/"index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Candidate images</title>"
        "<style>body{font:14px system-ui;padding:24px;}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}"
        "img{width:100%;height:195px;object-fit:contain;}article{border:1px solid #ddd;padding:8px}small{overflow-wrap:anywhere}</style>"
        "<h1>Leftover-plate candidate images</h1><p>CC BY 4.0; unannotated; many views per physical meal; not school-captured.</p>"
        "<main>" + "".join(
            f"<article><img loading='lazy' src='{html.escape(r['filename'],quote=True)}'><small>{html.escape(r['capture_group'])} | source row {r['source_row']}</small></article>"
            for r in rows
        ) + "</main>"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--limit",type=int,default=215)
    a=p.parse_args()
    run(a.output,a.limit)
