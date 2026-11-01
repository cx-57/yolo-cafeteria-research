"""Stage a provisional, class-balanced public GitHub photo review batch.

No object ground truth, independence, or disposal eligibility is claimed.
Source folders are hints and require individual visual inspection.
"""
import argparse
import csv
import hashlib
import io
import json
import random
import os
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
from PIL import Image, ImageDraw, ImageOps

REPO = "nikhilvenkatkumsetty/TrashBox"
REV = "main"
SOURCE = f"https://github.com/{REPO}"
SUBFOLDERS = {
    "bottle": "TrashBox_train_dataset_subfolders/plastic/plastic bottles/",
    "drink_carton": "TrashBox_train_dataset_subfolders/paper/tetra pak/",
    "container_unspecified": "TrashBox_train_dataset_subfolders/plastic/plastic containers/",
}

def download(url, token=None):
    headers = {"User-Agent": "cafeteria-research-provisional-source-review/1.0"}
    if token and url.startswith("https://api.github.com/"):
        headers["Authorization"] = "Bearer " + token
    with urlopen(Request(url, headers=headers), timeout=50) as r:
        return r.read()

def stage(out, quota=16, seed=20261101):
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite: {out}")
    out.mkdir(parents=True)
    (out/"images").mkdir()
    tree_url = f"https://api.github.com/repos/{REPO}/git/trees/{REV}?recursive=1"
    tree = json.loads(download(tree_url, os.environ.get("GH_TOKEN")))
    if tree.get("truncated"):
        raise RuntimeError("Source file tree was truncated; selection invalid")
    files = [r["path"] for r in tree.get("tree", [])
             if r.get("type")=="blob" and r["path"].lower().endswith((".jpg",".jpeg",".png",".webp"))]
    rng = random.Random(seed)
    rows = []
    seen = set()
    for kind, prefix in SUBFOLDERS.items():
        pool = [p for p in files if p.startswith(prefix)]
        rng.shuffle(pool)
        successes = 0
        for path in pool:
            if successes >= quota:
                break
            try:
                raw = download(f"https://raw.githubusercontent.com/{REPO}/{REV}/" + quote(path, safe="/"))
                if len(raw)>10_000_000 or len(raw)<800:
                    continue
                with Image.open(io.BytesIO(raw)) as im:
                    im.verify()
                with Image.open(io.BytesIO(raw)) as im:
                    width, height = im.size
                    if min(width,height)<90:
                        continue
                h=hashlib.sha256(raw).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                filename=f"{kind}_{successes+1:02d}_{h[:10]}" + Path(path).suffix.lower()
                (out/"images"/filename).write_bytes(raw)
                rows.append({
                    "filename":filename, "original_path":path, "sha256":h,
                    "source":SOURCE,"source_hint":kind,
                    "visually_verified":"NO","visual_status":"PENDING","visual_reason":"",
                    "historical_overlap":"NOT_CHECKED","independence":"UNRESOLVED",
                    "human_object_labels":"NONE","human_disposal_labels":"NONE",
                    "license":"NOT_VERIFIED",
                    "width":width,"height":height
                })
                successes += 1
            except Exception as exc:
                print("SKIP",path, str(exc)[:100],flush=True)
        print(f"{kind}: {successes} readable images of requested {quota}",flush=True)
    if not rows:
        raise RuntimeError("No candidate images downloaded")
    with (out/"review.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ncol=6; w=210; h=188; head=70
    sheet=Image.new("RGB",(w*ncol,head+h*((len(rows)+ncol-1)//ncol)),"white")
    draw=ImageDraw.Draw(sheet)
    draw.text((12,10),"TRASHBOX: CANDIDATES ONLY - no visual review, provenance or ground truth",fill="black")
    draw.text((12,32),"Folder names are source hints; verify license and class membership before use",fill="black")
    for i,row in enumerate(rows):
        x=(i%ncol)*w;y=head+(i//ncol)*h
        with Image.open(out/"images"/row["filename"]) as im:
            thumb=ImageOps.exif_transpose(im).convert("RGB")
            thumb.thumbnail((w-12,h-41))
        sheet.paste(thumb,(x+(w-thumb.width)//2,y))
        draw.text((x+5,y+h-34),row["filename"][:26],fill="black")
        draw.text((x+5,y+h-17),row["source_hint"][:29],fill="black")
    sheet.save(out/"contact-sheet.jpg",quality=87)
    summary=dict(status="UNVERIFIED_SOURCE_REVIEW",requested=quota*len(SUBFOLDERS),
                 staged=len(rows),
                 source_hint_counts={k:sum(r["source_hint"]==k for r in rows) for k in SUBFOLDERS},
                 visually_reviewed=0, independently_verified=0, human_bin_labels=0,
                 source=SOURCE,
                 limitations=[
                    "Internet GitHub public repository image folder names do not prove project class labels.",
                    "Plastic containers are not necessarily hinged cafeteria containers.",
                    "The source repository's image rights/redistribution permission have not been independently verified.",
                    "Images are single-object waste photos, not an independent school-cafeteria benchmark.",
                    "No local history or capture-group screening has been performed."
                 ])
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (out/"README.txt").write_text(
        "PROVISIONAL VISUAL REVIEW ONLY. No model inference, automatic bin ground truth or independent result.\n"
        "Photo source: "+SOURCE+"\n"
        "Source folders used: "+repr(SUBFOLDERS)+"\n"
        "Check source image rights before redistribution. Do not label container_unspecified as hinged_container without inspection.\n"
        "Check duplicates and all historical model/data exposure before final cohort admission.\n")
    print(json.dumps(summary, indent=2),flush=True)
    return summary

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--per-group",type=int,default=16)
    p.add_argument("--seed",type=int,default=20261101)
    args=p.parse_args()
    if args.per_group<1 or args.per_group>40:
        p.error("--per-group must be between 1 and 40")
    stage(args.out,args.per_group,args.seed)
