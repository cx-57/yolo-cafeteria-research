"""Isolated YOLO11m utensil specialist using V4 cafeteria photos and modest augmentation.

Does not overwrite frozen V1 or trained V4. Unreviewed labels remain provisional.
"""
import argparse,hashlib,json,os
from pathlib import Path
import yaml

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()

def make_dataset(root,out,legacy):
    rows=json.loads((root/"source_manifest.json").read_text())
    selected=["train_cafeteria","val"]+(["train_legacy"] if legacy else [])
    expected={"train_cafeteria":(521,294),"val":(97,48),"train_legacy":(161,244)}
    stats={}
    for src_split in selected:
        items=[r for r in rows if r["split"]==src_split]
        if len(items)!=expected[src_split][0]:
            raise ValueError(f"Changed {src_split} photo count")
        split="val" if src_split=="val" else "train"
        (out/"images"/split).mkdir(parents=True,exist_ok=True)
        (out/"labels"/split).mkdir(parents=True,exist_ok=True)
        box_count=0
        for record in items:
            image=root/record["image_path"]
            label=root/record["label_path"]
            if sha(image)!=record["image_sha256"] or sha(label)!=record["label_sha256"]:
                raise ValueError(f"Modified image or label: {image}")
            # Hardlink source image; do not edit original pixels.
            name=src_split+"__"+image.stem
            os.link(image,out/"images"/split/(name+image.suffix.lower()))
            lines=[]
            for line in label.read_text().splitlines():
                if not line.strip():continue
                parts=line.split()
                if int(parts[0])==2:
                    parts[0]="0"
                    lines.append(" ".join(parts))
            box_count+=len(lines)
            (out/"labels"/split/(name+".txt")).write_text(
                "\n".join(lines)+("\n" if lines else ""))
        if box_count!=expected[src_split][1]:
            raise ValueError(f"Utensil annotations changed in {src_split}")
        stats[src_split]={"images":len(items),"utensil_boxes":box_count}
    cfg={"path":str(out.resolve()),"train":"images/train","val":"images/val",
         "names":["utensil"]}
    (out/"data.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False))
    return out/"data.yaml",stats

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--include-legacy",action="store_true")
    p.add_argument("--device",default="0")
    p.add_argument("--batch",type=int,default=4)
    args=p.parse_args()
    if args.output.exists():p.error("Output directory already exists")
    if args.batch<1:p.error("Invalid batch")
    args.output.mkdir(parents=True)
    data,counts=make_dataset(args.root,args.output/"single_class",args.include_legacy)
    (args.output/"source_counts.json").write_text(json.dumps(counts,indent=2)+"\n")
    from ultralytics import YOLO
    print("Training 20-epoch YOLO11m utensil specialist:",counts,flush=True)
    m=YOLO("yolo11m.pt")
    m.train(data=str(data),epochs=20,patience=10,imgsz=800,batch=args.batch,
            device=args.device,workers=2,optimizer="AdamW",lr0=.0005,lrf=.01,
            weight_decay=.0005,seed=42,deterministic=True,cache=False,
            degrees=10,translate=.08,scale=.25,
            hsv_h=.015,hsv_s=.35,hsv_v=.25,
            fliplr=.5,flipud=0,mosaic=.5,mixup=0,close_mosaic=10,
            project=str(args.output),name="train",exist_ok=False,plots=True)
    best=args.output/"train/weights/best.pt"
    if not best.is_file():raise FileNotFoundError(best)
    met=YOLO(str(best)).val(data=str(data),split="val",conf=.001,iou=.7,
                           imgsz=800,device=args.device,batch=args.batch,plots=False)
    record={"model":"YOLO11m utensil-only","pretrained":"yolo11m.pt",
            "epochs_max":20,"imgsz":800,"include_legacy":args.include_legacy,
            "val_images":97,"val_annotations":48,
            "AP50":float(met.box.map50),"AP50_95":float(met.box.map),
            "best_sha256":sha(best),"test_evaluated":False,
            "annotation_human_review_complete":False}
    (args.output/"utensil_val_metrics.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record,indent=2),flush=True)

if __name__=="__main__":main()
