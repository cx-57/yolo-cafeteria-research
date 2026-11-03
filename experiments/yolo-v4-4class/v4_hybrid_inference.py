"""V4 food/plate/wrapper + frozen V1 utensil specialist. Inference only."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2

NAMES = ["food", "plate", "utensil", "wrapper"]
UTENSIL_SHA256 = "ce4956547f0c421db1dc4b3d2fa7c28d9d9ae2fc44ddf5acf333ff211c3d4c7f"
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def names(model):
    n = model.names
    return [n.get(i, n.get(str(i))) for i in range(len(n))] if isinstance(n, dict) else list(n)

def boxes(result, mapping):
    if result.boxes is None:
        return []
    found = []
    for xy, conf, cls in zip(result.boxes.xyxy.cpu().tolist(),
                             result.boxes.conf.cpu().tolist(),
                             result.boxes.cls.cpu().tolist()):
        c = int(cls)
        if c not in mapping:
            continue
        mapped = mapping[c]
        found.append({"class_id": mapped, "class": NAMES[mapped],
                      "confidence": round(float(conf), 5),
                      "xyxy": [round(float(v), 1) for v in xy]})
    return found

def draw(image, found):
    out = image.copy()
    colors = {"food": (55,180,60), "plate": (215,110,30),
              "utensil": (45,65,220), "wrapper": (190,55,155)}
    for row in found:
        x1,y1,x2,y2 = map(lambda v:int(round(v)), row["xyxy"])
        color = colors[row["class"]]
        cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
        cv2.putText(out,f'{row["class"]} {row["confidence"]:.2f}',
                    (x1,max(16,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,2,cv2.LINE_AA)
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v4",type=Path,required=True)
    p.add_argument("--utensil",type=Path,required=True)
    p.add_argument("--images",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="auto")
    p.add_argument("--conf",type=float,default=.25)
    args=p.parse_args()
    if not 0<=args.conf<=1:
        p.error("Confidence must be in [0,1]")
    for file in [args.v4,args.utensil]:
        if not file.is_file():
            p.error(f"Checkpoint not found: {file}")
    if sha(args.utensil)!=UTENSIL_SHA256:
        p.error("Wrong utensil checkpoint: expected original frozen V1 weights/utensil.pt")
    if args.output.exists():
        p.error(f"Refusing to overwrite {args.output}")
    if args.images.is_file():
        images=[args.images]
    elif args.images.is_dir():
        images=sorted(x for x in args.images.iterdir() if x.is_file() and x.suffix.lower() in SUFFIXES)
    else:
        p.error(f"Missing images: {args.images}")
    if not images or len({x.stem for x in images})!=len(images):
        p.error("Need photos with distinct filenames")
    from ultralytics import YOLO
    v4=YOLO(str(args.v4))
    specialist=YOLO(str(args.utensil))
    if names(v4)!=NAMES or names(specialist)!=["utensil"]:
        p.error(f"Wrong class mapping: V4={v4.names}, specialist={specialist.names}")
    device=None if args.device=="auto" else args.device
    args.output.mkdir(parents=True)
    records=[]
    for i,path in enumerate(images,1):
        image=cv2.imread(str(path))
        if image is None:
            print(f"Unreadable image, skipped: {path}",flush=True)
            continue
        v4_out=v4.predict(image,imgsz=800,conf=args.conf,iou=.7,device=device,verbose=False)[0]
        specialist_out=specialist.predict(image,imgsz=800,conf=args.conf,iou=.7,device=device,verbose=False)[0]
        base=boxes(v4_out,{0:0,1:1,2:2,3:3})
        expert=boxes(specialist_out,{0:2})
        hybrid=[b for b in base if b["class"]!="utensil"]+expert
        cv2.imwrite(str(args.output/(path.stem+"_v4.jpg")),draw(image,base))
        cv2.imwrite(str(args.output/(path.stem+"_hybrid.jpg")),draw(image,hybrid))
        records.append({"file":path.name,"v4":base,"hybrid":hybrid})
        print(f'{i}/{len(images)} {path.name}: V4 utensils={sum(b["class"]=="utensil" for b in base)}; specialist={len(expert)}',flush=True)
    (args.output/"detections.json").write_text(json.dumps({
        "warning":"Unlabeled visual comparison, not measured accuracy",
        "v4_sha256":sha(args.v4),"utensil_sha256":sha(args.utensil),
        "confidence":args.conf,"images":records},indent=2)+"\n")
    print(f"Saved {len(records)} comparison pairs to {args.output}",flush=True)

if __name__=="__main__":
    main()
