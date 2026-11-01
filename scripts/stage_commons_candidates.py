"""Download openly licensed Wikimedia Commons photos for *manual* cafeteria-object screening.

This script does not verify object classes, source independence, or bin accuracy.
It preserves per-image creator, license and source URLs; it does not run YOLO.
"""
import argparse
import csv
import hashlib
import io
import json
import random
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageOps

API = "https://commons.wikimedia.org/w/api.php"
SOURCES = {
    "utensil": ["Plastic cutlery", "Disposable plastic forks", "Disposable plastic spoons"],
    "wrapper": ["Candy wrappers"],
    "hinged_container_candidate": ["Expanded polystyrene food boxes", "Disposable lunch boxes"],
}

def get(url):
    headers = {"User-Agent": "CafeteriaResearchImageReview/0.1 (educational research; requests=source-photo-screening)",
               "Accept": "application/json, image/*;q=0.9,*/*;q=0.5"}
    with urlopen(Request(url, headers=headers), timeout=45) as response:
        return response.read()

def category_media(category):
    params = {
        "action":"query", "generator":"categorymembers",
        "gcmtitle":"Category:" + category, "gcmtype":"file", "gcmlimit":"100",
        "prop":"imageinfo", "iiprop":"url|size|extmetadata", "iiurlwidth":"1000",
        "format":"json", "formatversion":"2",
    }
    payload = json.loads(get(API + "?" + urlencode(params)))
    if "error" in payload:
        raise RuntimeError("Wikimedia API: " + str(payload["error"])[:300])
    return payload.get("query", {}).get("pages", [])

def is_licensed(metadata):
    license_short = metadata.get("LicenseShortName", {}).get("value", "")
    allowed = ("CC BY", "CC0", "Public domain", "PD-", "Public Domain")
    return license_short.startswith(allowed), license_short

def build_review(out, target_each, seed):
    if out.exists():
        raise FileExistsError("Output exists: refusing to overwrite " + str(out))
    out.mkdir(parents=True)
    (out/"images").mkdir()
    records=[]; keys=set(); hset=set(); rng=random.Random(seed)
    for label, categories in SOURCES.items():
        candidates=[]
        for cat in categories:
            try:
                pages=category_media(cat)
                for page in pages:
                    for ii in page.get("imageinfo", []):
                        extra=ii.get("extmetadata", {})
                        license_ok,lic=is_licensed(extra)
                        url=ii.get("thumburl") or ii.get("url")
                        if not license_ok or not url or not ii.get("mime", "").startswith("image/"):
                            continue
                        if page.get("title") in keys:
                            continue
                        if ii.get("width",0)<350 or ii.get("height",0)<300:
                            continue
                        candidates.append((page,ii,cat,lic))
                        keys.add(page.get("title"))
                print(f"{cat}: {len(pages)} category pages, {len(candidates)} cumulative licensed candidates",flush=True)
            except Exception as e:
                print(f"ERROR source category {cat}: {e}",flush=True)
        rng.shuffle(candidates)
        accepted=0
        for page,ii,cat,lic in candidates:
            if accepted>=target_each:
                break
            try:
                raw=get(ii.get("thumburl") or ii["url"])
                if len(raw)>10_000_000 or len(raw)<2000:
                    continue
                with Image.open(io.BytesIO(raw)) as im:
                    im.verify()
                with Image.open(io.BytesIO(raw)) as im:
                    W,H=im.size
                    if min(W,H)<250:
                        continue
                sha=hashlib.sha256(raw).hexdigest()
                if sha in hset:
                    continue
                hset.add(sha)
                ext=".jpg" if "jpeg" in ii.get("mime","") else ".png"
                name=f"{label}_{accepted+1:03}_{sha[:11]}{ext}"
                (out/"images"/name).write_bytes(raw)
                meta=ii.get("extmetadata",{})
                records.append({
                    "filename":name, "project_class_hint":label,
                    "source_category":cat,"commons_title":page["title"],
                    "commons_page":ii.get("descriptionurl",""),
                    "original_url":ii.get("url",""),
                    "download_url":ii.get("thumburl") or ii.get("url"),
                    "creator_html":meta.get("Artist",{}).get("value",""),
                    "license":lic,
                    "license_url":meta.get("LicenseUrl",{}).get("value",""),
                    "attribution_html":meta.get("Attribution",{}).get("value",""),
                    "sha256":sha,"width":W,"height":H,
                    "visual_review":"PENDING","class_confirmed":"NO",
                    "historical_overlap":"NOT_CHECKED",
                    "independent":"NO","human_ground_truth":"NONE",
                    "bin_ground_truth":"NONE"})
                accepted+=1
            except Exception as e:
                print("SKIP", page.get("title"),str(e)[:150],flush=True)
        print(f"{label}: staged {accepted}",flush=True)
    if not records:
        raise RuntimeError("Wikimedia failed to supply any usable image files")
    with (out/"review.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=records[0].keys());w.writeheader();w.writerows(records)
    cols=6;sw=210;sh=174;head=68
    preview=Image.new("RGB",(cols*sw,head+((len(records)+cols-1)//cols)*sh),"white")
    d=ImageDraw.Draw(preview)
    d.text((10,8),"Wikimedia Commons SOURCE CANDIDATES — NOT visually reviewed or independently certified",fill="black")
    d.text((10,32),"License per-image in review.csv; creator/source attribution required",fill="black")
    for k,row in enumerate(records):
        x=k%cols*sw;y=head+(k//cols)*sh
        with Image.open(out/"images"/row["filename"]) as original:
            im=ImageOps.exif_transpose(original).convert("RGB")
            im.thumbnail((sw-8,sh-37))
        preview.paste(im,(x+(sw-im.width)//2,y))
        d.text((x+3,y+sh-32),row["filename"][:25],fill="black")
        d.text((x+3,y+sh-16),row["source_category"][:24],fill="black")
    preview.save(out/"contact-sheet.jpg",quality=88)
    (out/"README.md").write_text(
        "# Wikimedia image-screening pool — NOT independent evaluation\n\n"
        "The images were downloaded through the Wikimedia Commons API from specific source categories. "
        "Only per-image CC BY / CC BY-SA / CC0 / public-domain license claims were accepted by software. "
        "Verify each image's licensing page and attribution requirements before redistribution/publication.\n\n"
        "The source category is NOT project-class ground truth; individually inspect each photo. "
        "In particular, many foam food boxes may not be hinged. This program does not annotate objects, "
        "determine school bins, check historical exposure or run the frozen detectors.\n")
    summary=dict(fetched=len(records),counts={k:sum(row["project_class_hint"]==k for row in records) for k in SOURCES},
                 visual_reviews=0,independent_verified=0,bin_ground_truth=0,
                 frozen_inference_run=False)
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--per-group",default=16,type=int)
    p.add_argument("--seed",default=20261101,type=int)
    a=p.parse_args()
    if not 1<=a.per_group<=30:p.error("per-group must be 1..30")
    build_review(a.out,a.per_group,a.seed)
