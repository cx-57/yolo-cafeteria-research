"""Validation-only Ultralytics AP for V4, V4+V1 utensil, and simple fusion.
Uses the unchanged V4 97-image validation split. Not an independent field test.
Run after extracting the V4 dataset ZIP and providing BOTH model checkpoints.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import cv2

NAMES=["food","plate","utensil","wrapper"]
V4_SHA="0be9895e6c60bfe0b5fd478b01b6aef6484c24f827bf20ea7e65e7e34f66f343"
UT_SHA="ce4956547f0c421db1dc4b3d2fa7c28d9d9ae2fc44ddf5acf333ff211c3d4c7f"
THRESHOLDS=np.linspace(.5,.95,10)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def pred_matrix(result,utensil=False):
    if result.boxes is None or not len(result.boxes):
        return np.zeros((0,6),dtype=np.float32)
    b=result.boxes
    xy=b.xyxy.cpu().numpy().astype(np.float32)
    co=b.conf.cpu().numpy().reshape(-1,1).astype(np.float32)
    cl=b.cls.cpu().numpy().reshape(-1,1).astype(np.float32)
    if utensil:
        if np.any(cl!=0):raise ValueError("Specialist must be single-class utensil")
        cl[:]=2
    return np.concatenate([xy,co,cl],axis=1)

def ground_truth(path,w,h):
    a=[]
    for line in path.read_text().splitlines():
        if not line.strip():continue
        c,x,y,bw,bh=map(float,line.split())
        a.append([c,(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
    return np.asarray(a,dtype=np.float32).reshape(-1,5)

def iou(a,b):
    if not len(a) or not len(b):return np.zeros((len(a),len(b)),dtype=np.float32)
    upper=np.minimum(a[:,None,2:4],b[None,:,2:4])
    lower=np.maximum(a[:,None,:2],b[None,:,:2])
    inter=np.maximum(upper-lower,0).prod(2)
    area_a=np.maximum(a[:,2:4]-a[:,:2],0).prod(1)
    area_b=np.maximum(b[:,2:4]-b[:,:2],0).prod(1)
    return inter/(area_a[:,None]+area_b[None,:]-inter+1e-8)

def matches(p,gt):
    result=np.zeros((len(p),10),dtype=bool)
    if not len(p) or not len(gt):return result
    overlap=iou(gt[:,1:5],p[:,:4])
    overlap[gt[:,0,None]!=p[None,:,5]]=0
    for j,threshold in enumerate(THRESHOLDS):
        candidates=np.argwhere(overlap>=threshold)
        candidates=candidates[np.argsort(-overlap[candidates[:,0],candidates[:,1]],kind="stable")]
        used_gt=set();used_pred=set()
        for g,k in candidates:
            if int(g) in used_gt or int(k) in used_pred:continue
            result[k,j]=True;used_gt.add(int(g));used_pred.add(int(k))
    return result

def selective_fusion(base,expert):
    """Specialist utensils plus V4 utensil confidence >=.50; NMS IoU .55."""
    non_ut=base[base[:,5]!=2]
    addon=base[(base[:,5]==2)&(base[:,4]>=.5)]
    joined=np.concatenate([expert,addon])
    joined=joined[np.argsort(-joined[:,4])]
    kept=[]
    for item in joined:
        if not kept or np.all(iou(item[None,:4],np.asarray(kept)[:,:4])[0]<.55):
            kept.append(item)
    return np.concatenate([non_ut,np.asarray(kept,dtype=np.float32).reshape(-1,6)])

def scores(log):
    from ultralytics.utils.metrics import ap_per_class
    tp=np.concatenate([a[0] for a in log])
    conf=np.concatenate([a[1] for a in log])
    pred_cls=np.concatenate([a[2] for a in log])
    target=np.concatenate([a[3] for a in log])
    result=ap_per_class(tp,conf,pred_cls,target,plot=False)
    p,r,ap,classes=result[2],result[3],result[5],result[6]
    per={}
    for i,c in enumerate(classes):
        per[NAMES[int(c)]]={"AP50":round(float(ap[i,0]),6),
                           "AP50_95":round(float(ap[i].mean()),6),
                           "annotations":int((target==c).sum())}
    return {"mAP50":round(float(ap[:,0].mean()),6),
            "mAP50_95":round(float(ap.mean()),6),
            "precision_maxF1":round(float(p.mean()),6),
            "recall_maxF1":round(float(r.mean()),6),"per_class":per}

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--root",type=Path,required=True)
    a.add_argument("--v4",type=Path,required=True)
    a.add_argument("--utensil",type=Path,required=True)
    a.add_argument("--output",type=Path,required=True)
    a.add_argument("--device",default="cpu")
    x=a.parse_args()
    if x.output.exists():a.error("Output folder already exists")
    if sha(x.v4)!=V4_SHA or sha(x.utensil)!=UT_SHA:
        a.error("Unexpected model checkpoint SHA256")
    rows=[r for r in json.loads((x.root/"source_manifest.json").read_text()) if r["split"]=="val"]
    if len(rows)!=97:a.error("Expected exactly 97 V4 validation photos")
    from ultralytics import YOLO
    from v4_hybrid_inference import repair_legacy_head_pooling
    v4=YOLO(str(x.v4));ut=YOLO(str(x.utensil))
    def names(m):
        return [m.names.get(i,m.names.get(str(i))) for i in range(len(m.names))] if isinstance(m.names,dict) else list(m.names)
    if names(v4)!=NAMES or names(ut)!=["utensil"]:a.error("Model class mapping mismatch")
    repair_legacy_head_pooling(ut)
    logs={k:[] for k in ["v4","hybrid","fusion"]}
    seen_boxes=np.zeros(4,dtype=int)
    for index,row in enumerate(rows,1):
        imp=x.root/row["image_path"];lb=x.root/row["label_path"]
        if sha(imp)!=row["image_sha256"] or sha(lb)!=row["label_sha256"]:
            a.error("Validation images or labels changed")
        image=cv2.imread(str(imp))
        if image is None:raise ValueError(f"Cannot decode {imp}")
        h,w=image.shape[:2];gt=ground_truth(lb,w,h)
        for c in range(4):seen_boxes[c]+=int((gt[:,0]==c).sum())
        opts=dict(imgsz=800,conf=.001,iou=.7,max_det=300,device=x.device,verbose=False,augment=False)
        base=pred_matrix(v4.predict(image,**opts)[0])
        spec=pred_matrix(ut.predict(image,**opts)[0],utensil=True)
        variants={"v4":base,
                  "hybrid":np.concatenate([base[base[:,5]!=2],spec]),
                  "fusion":selective_fusion(base,spec)}
        for key,p in variants.items():
            logs[key].append((matches(p,gt),p[:,4],p[:,5],gt[:,0]))
        if index%10==0 or index==97:print(f"Processed {index}/97 images",flush=True)
    if seen_boxes.tolist()!=[274,163,48,13]:
        a.error(f"Unexpected target counts: {seen_boxes.tolist()}")
    report={"split":"val","images":97,"labels_audited_by_human":False,
            "test_evaluated":False,"conf":.001,
            "metric":"Ultralytics ap_per_class over 10 IoU thresholds",
            "fusion":"Specialist + V4 utensil conf>=.50, utensil-only NMS IoU .55",
            "v4_sha256":sha(x.v4),"utensil_sha256":sha(x.utensil),
            "results":{k:scores(v) for k,v in logs.items()}}
    x.output.mkdir(parents=True)
    (x.output/"hybrid_val_map.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report["results"],indent=2),flush=True)

if __name__=="__main__":main()
