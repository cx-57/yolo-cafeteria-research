#!/usr/bin/env python3
"""Separate 865-image seven-class YOLO11n legacy-supplement trial. Never edits V1 specialists."""
import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

NAMES = ["food", "plate", "utensil", "hinged_container", "drink_carton", "wrapper", "bottle"]
HERE = Path(__file__).resolve().parent

def checksum(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def verify_dataset(folder):
    """Verify all image hashes, label hashes, class geometry, and source-group split isolation."""
    from PIL import Image
    path = folder / "source_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing source manifest: {path}")
    records = json.loads(path.read_text())
    if len(records) != 865:
        raise ValueError(f"Expected 865 images, got {len(records)}")
    splits = Counter()
    groups, seen_images = {}, set()
    objects = defaultdict(Counter)
    for r in records:
        split, name, group = r["split"], r["output_name"], r["group"]
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split {split}")
        image = folder / "images" / split / name
        label = folder / "labels" / split / (Path(name).stem + ".txt")
        if not image.exists() or not label.exists():
            raise FileNotFoundError(f"Missing paired image/label: {image}")
        if checksum(image) != r["image_sha256"] or checksum(label) != r["label_sha256"]:
            raise ValueError(f"Source hashes changed for {name}")
        if r["image_sha256"] in seen_images:
            raise ValueError(f"Duplicate image bytes: {name}")
        seen_images.add(r["image_sha256"])
        if group in groups and groups[group] != split:
            raise ValueError(f"Group leakage: {group}")
        groups[group] = split
        with Image.open(image) as im:
            im.verify()
        for row in label.read_text().splitlines():
            if not row.strip():
                continue
            vals = [float(x) for x in row.split()]
            if len(vals) != 5 or not all(math.isfinite(x) for x in vals):
                raise ValueError(f"Invalid YOLO label in {label}")
            cls, x, y, w, h = vals
            if (cls != int(cls) or not 0 <= cls < 7 or w <= 0 or h <= 0
                or x-w/2 < -1e-6 or y-h/2 < -1e-6
                or x+w/2 > 1+1e-6 or y+h/2 > 1+1e-6):
                raise ValueError(f"Bad bounding box: {label} {row}")
            objects[split][int(cls)] += 1
        splits[split] += 1
    expected = {"train":697, "val":97, "test":71}
    if dict(splits) != expected:
        raise ValueError(f"Split counts changed: {splits}")
    expected_classes = {0:2266, 1:1130, 2:621, 3:0, 4:0, 5:140, 6:0}
    actual_classes = {i:sum(objects[split][i] for split in ("train","val","test")) for i in range(7)}
    if actual_classes != expected_classes:
        raise ValueError(f"Class object counts changed: {actual_classes}")
    total = sum(sum(counts.values()) for counts in objects.values())
    if total != 4157:
        raise ValueError(f"Expected 4157 boxes, found {total}")
    return {"images":dict(splits), "source_groups":len(groups),
            "boxes":total,
            "per_split_boxes":{s:{NAMES[i]:objects[s][i] for i in range(7)}
                               for s in ("train","val","test")},
            "manifest_sha256":checksum(path)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=HERE/"yolo_v3_865_legacy"/"data.yaml")
    parser.add_argument("--output",type=Path,default=HERE/"YOLO_V3_trial_results")
    parser.add_argument("--model",default="yolo11n.pt")
    parser.add_argument("--device",default="auto")
    parser.add_argument("--epochs",type=int,default=20)
    parser.add_argument("--batch",type=int,default=4)
    parser.add_argument("--imgsz",type=int,default=640)
    parser.add_argument("--verify-only",action="store_true")
    args = parser.parse_args()
    data_path = args.data.expanduser().resolve()
    if not data_path.is_file():
        parser.error(f"Dataset missing: {data_path}. Extract the ZIP first.")
    if args.batch < 1 or args.epochs < 1 or args.imgsz < 128 or args.imgsz % 32:
        parser.error("Invalid epochs/batch/imgsz")
    folder = data_path.parent
    checks = verify_dataset(folder)
    print(json.dumps(checks, indent=2), flush=True)
    if args.verify_only:
        return

    import torch
    import ultralytics
    import yaml
    from ultralytics import YOLO
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else ("0" if torch.cuda.is_available() else "cpu")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Results already exist, refusing overwrite: {output}")
    output.mkdir(parents=True)
    config_path = output/"resolved_data.yaml"
    config_path.write_text(yaml.safe_dump(dict(
        path=str(folder),train="images/train",val="images/val",test="images/test",
        names=NAMES,nc=7),sort_keys=False))
    (output/"experiment_config.json").write_text(json.dumps(dict(
        experiment="isolated_yolo11n_865_legacy_supplement",date_utc=datetime.now(timezone.utc).isoformat(),
        starting_model=args.model,model="YOLO11n",training_images=697,
        device=device,epochs=args.epochs,batch=args.batch,imgsz=args.imgsz,
        ultralytics=ultralytics.__version__,torch=torch.__version__,
        data=checks,
        warning="Mixed provisional AI-reviewed and recovered source annotations; only one missing fork manually added; classes 3, 4, 6 absent; not an independent field test."
    ),indent=2))
    model = YOLO(args.model)
    if model.task != "detect":
        raise ValueError("Detection model required")
    model.train(data=str(config_path),epochs=args.epochs,imgsz=args.imgsz,
        batch=args.batch,device=device,workers=0,seed=42,deterministic=True,
        patience=10,project=str(output),name="train",exist_ok=False,
        optimizer="AdamW",lr0=0.0005,weight_decay=0.0005,
        cache=False,plots=True,save=True,mosaic=0.5,close_mosaic=10,
        mixup=0.0,degrees=10.0,translate=0.06,scale=0.2,fliplr=0.5,
        hsv_h=0.015,hsv_s=0.3,hsv_v=0.2)
    best = output/"train"/"weights"/"best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Missing trained weights: {best}")
    result = YOLO(str(best)).val(
        data=str(config_path),split="test",imgsz=args.imgsz,batch=args.batch,
        device=device,workers=0,plots=True,
        project=str(output),name="test",exist_ok=False)
    support = checks["per_split_boxes"]["test"]
    metrics = {}
    for i,c in enumerate(map(int,result.box.ap_class_index)):
        if 0 <= c < 7:
            metrics[NAMES[c]] = {"test_boxes":support[NAMES[c]],
                "ap50":float(result.box.ap50[i]),
                "ap50_95":float(result.box.ap[i].mean())}
    summary = dict(
        test_images=71,
        mAP50=float(result.box.map50),mAP50_95=float(result.box.map),
        precision=float(result.box.mp),recall=float(result.box.mr),
        per_class=metrics,
        unsupported_classes=[n for n in NAMES if not support[n]],
        best_weights=str(best),best_weights_sha256=checksum(best),
        caution="Group-held-out historically exposed exploratory metrics; one test fork label corrected; no independent field generalization")
    (output/"test_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":
    main()
