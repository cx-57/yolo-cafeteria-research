"""Stage openly licensed Wikimedia Commons waste/cafeteria scene photographs.

These are SOURCE-REVIEW CANDIDATES, never seven-class YOLO labels.
Respect source attribution/license, exclude images with ambiguous rights.
"""
import argparse, csv, hashlib, html, io, json, random, time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from PIL import Image, ImageDraw, ImageOps

API="https://commons.wikimedia.org/w/api.php"
CATEGORIES=[
    "Food waste", "Food leftovers", "School lunches", "School meals",
    "Cafeterias", "Meal trays", "Disposable food containers",
    "Disposable plates", "Fast food packaging", "Litter",
    "Plastic bottles", "Composting",
]
ALLOWED=("CC BY", "CC0", "Public domain", "Public Domain", "PD-")


def http(url, accept="application/json"):
    headers={"User-Agent":"CafeteriaWasteResearch/1.0 (student research, Commons license-respecting review; noncommercial)",
             "Accept":accept}
    with urlopen(Request(url,headers=headers),timeout=35) as response:
        return response.read()


def list_files(category):
    query=dict(action="query",generator="categorymembers",gcmtitle="Category:"+category,
               gcmtype="file",gcmlimit="100",prop="imageinfo",
               iiprop="url|size|mime|extmetadata",iiurlwidth="1280",format="json",formatversion="2")
    data=json.loads(http(API+"?"+urlencode(query)))
    return [p for p in data.get("query",{}).get("pages",[]) if p.get("imageinfo")]


def main(output,limit=110,per_category=15):
    root=Path(output).resolve()
    if root.exists(): raise FileExistsError(root)
    (root/"images").mkdir(parents=True)
    records=[]; seen=set(); reject_log=[]
    for category in CATEGORIES:
        if len(records)>=limit: break
        try:
            candidates=list_files(category)
        except Exception as e:
            reject_log.append(dict(category=category,reason="QUERY_ERROR",detail=str(e)[:160]))
            continue
        kept=0
        for page in sorted(candidates,key=lambda p:p.get("title","")):
            if kept>=per_category or len(records)>=limit:break
            info=page["imageinfo"][0]
            ext=info.get("extmetadata",{})
            license_=ext.get("LicenseShortName",{}).get("value","")
            mime=info.get("mime","")
            if not license_.startswith(ALLOWED) or mime not in ("image/jpeg","image/png","image/webp"):
                continue
            if info.get("width",0)<420 or info.get("height",0)<300:continue
            title=page.get("title","")
            if title in seen:continue
            seen.add(title)
            uri=info.get("thumburl") or info.get("url")
            if not uri:continue
            try:
                blob=http(uri, "image/*")
                if not (3000<len(blob)<9_000_000):continue
                with Image.open(io.BytesIO(blob)) as photo:
                    photo.verify()
                with Image.open(io.BytesIO(blob)) as photo:
                    im=ImageOps.exif_transpose(photo).convert("RGB")
                if im.width<400 or im.height<280:continue
                # Preserve exact derived JPEG identity and source original URL.
                raw=io.BytesIO();im.save(raw,"JPEG",quality=90)
                data=raw.getvalue();digest=hashlib.sha256(data).hexdigest()
                filename=f"{len(records):04d}-{category.lower().replace(' ','_')}-{digest[:12]}.jpg"
                (root/"images"/filename).write_bytes(data)
                records.append(dict(
                    file_name="images/"+filename,sha256=digest,
                    source_category=category,source_title=title,
                    original_url=info.get("url",""),source_page=info.get("descriptionurl",""),
                    license=license_,license_url=ext.get("LicenseUrl",{}).get("value",""),
                    creator_html=ext.get("Artist",{}).get("value",""),
                    attribution_html=ext.get("Attribution",{}).get("value",""),
                    image_width=im.width,image_height=im.height,
                    research_relevance="NOT_REVIEWED",
                    seven_class_annotations_complete="NO",
                    school_domain="UNVERIFIED",
                    historical_overlap="NOT_CHECKED"
                ))
                kept+=1
            except Exception as e:
                reject_log.append(dict(category=category,reason="IMAGE_FETCH_FAILED",detail=str(e)[:150],image=title))
        print(f"Source category '{category}': {kept} downloads",flush=True)
    if not records:raise RuntimeError("No suitable licensed Commons photos fetched")
    with (root/"manifest.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (root/"acquisition_log.json").write_text(json.dumps(reject_log,indent=2)+"\n")
    summary=dict(status="UNANNOTATED_PHOTO_CANDIDATES_NEED_VISUAL_REVIEW",photos=len(records),
                 source="Wikimedia Commons",category_counts={c:sum(r["source_category"]==c for r in records) for c in CATEGORIES},
                 licenses=sorted(set(r["license"] for r in records)),human_reviewed_images=0,
                 yolo_7_class_training_ready=False,attribution_required=True)
    (root/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (root/"README.md").write_text(
        "# Wikimedia Commons cafeteria and waste image candidates\n\n"
        "These are source-category-screened, NOT human-verified relevant images and NOT complete seven-class YOLO labels. "
        "Source category names do not prove target objects are present. See manifest.csv for per-image source page, "
        "photographer/creator, exact license, license URL and source image URL. Some images may use CC BY-SA and "
        "derivatives may have ShareAlike obligations. Do not use images with watermarks, people/private details, "
        "unclear rights or irrelevant scenes. Inspect every photo and label all seven classes before adding to training. "
        "Check source duplicates and capture groups against local historical training/validation/test inventories.\n"
    )
    cols,w,h=6,205,175
    sheet=Image.new("RGB",(cols*w,((len(records)+cols-1)//cols)*h+54),"white")
    d=ImageDraw.Draw(sheet)
    d.text((10,15),"Wikimedia Commons SCREENING CANDIDATES — unverified class labels",fill="black")
    for i,row in enumerate(records):
        with Image.open(root/row["file_name"]) as orig:thumb=orig.copy()
        thumb.thumbnail((w-12,h-34))
        x=i%cols*w;y=(i//cols)*h+54
        sheet.paste(thumb,(x+(w-thumb.width)//2,y))
        d.text((x+4,y+h-22),row["source_category"][:23],fill="black")
    sheet.save(root/"contact-sheet.jpg",quality=85)
    (root/"index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Commons candidates</title>"
        "<style>body{font:14px system-ui;padding:24px}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}"
        "img{width:100%;height:190px;object-fit:contain}article{border:1px solid #ddd;padding:8px}</style>"
        "<h1>Original-source Wikimedia Commons candidate gallery</h1><p>Not annotated; verify relevance and license.</p><main>"
        +"".join("<article><img src='"+html.escape(r["file_name"],quote=True)+"'><p>"+
                 html.escape(r["source_category"])+" — <a href='"+html.escape(r["source_page"],quote=True)+
                 "'>Source and license</a></p></article>" for r in records)+"</main>")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    p.add_argument("--limit",type=int,default=110)
    p.add_argument("--per-category",type=int,default=15)
    a=p.parse_args()
    main(a.output,a.limit,a.per_category)
