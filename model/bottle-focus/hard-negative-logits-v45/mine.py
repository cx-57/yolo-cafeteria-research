"""Mine only reviewed negative TRAINING photos, never monitoring/benchmark images."""
from pathlib import Path
import hashlib
import json
import sys
import torch
from ultralytics import YOLO
from batches import predict_chunks
RUN=Path(__file__).resolve().parent;ROOT=RUN.parent;V44=ROOT/'classifier-retention-v44'
SOURCE=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-reviewed-neck-monitor-v36/weights/verified-88877bbf13c02a202811b6ca0a1ccdc2bde70b2916d366f4b67938df732d1518.pt')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
if __name__=='__main__':
 torch.set_num_threads(2);torch.mps.set_per_process_memory_fraction(.5);audit=json.loads((V44/'dataset.json').read_text());rows=[r for r in audit['training_records'] if not r['rows']]
 assert len(rows)==142 and sha(SOURCE)=='88877bbf13c02a202811b6ca0a1ccdc2bde70b2916d366f4b67938df732d1518'
 reserved=json.loads((V44/'source-selection.json').read_text())['old_monitor_records']+audit['retention_records']
 for key in ['id','image_sha','pixels','source_group']:assert not {r[key] for r in rows}&{r[key] for r in reserved}
 for r in rows:assert sha(r['image'])==r['image_sha'] and Path(r['label']).read_text().splitlines()==[]
 model=YOLO(str(SOURCE));records=[]
 for row,result in predict_chunks(model,rows):
  assert Path(result.path)==Path(row['image'])
  boxes=[dict(xyxy=b.xyxy[0].cpu().tolist(),score=float(b.conf[0])) for b in result.boxes]
  records.append(dict(row,predictions=boxes,maximum_false_detection_score=max([b['score'] for b in boxes],default=0.)))
  if len(records)%4==0:
   print('MINED_NEGATIVE_PHOTOS',len(records),flush=True);torch.mps.empty_cache()
 assert sha(SOURCE)=='88877bbf13c02a202811b6ca0a1ccdc2bde70b2916d366f4b67938df732d1518'
 report=dict(source_weights_sha=sha(SOURCE),source_dataset_sha=sha(V44/'dataset.json'),negative_training_photos=142,source='training_only_no_monitoring_or_benchmark_inference',mining_confidence=.1,nms_iou=.5,imgsz=640,batch=4,records=records)
 (RUN/'mining.json').write_text(json.dumps(report,indent=2)+'\n')
 hard=sorted([r for r in records if r['predictions']],key=lambda r:(-r['maximum_false_detection_score'],r['id']))
 (RUN/'hard-candidates.json').write_text(json.dumps(hard[:16],indent=2)+'\n')
 print('HARD_NEGATIVE_MINING',len(rows),'photos; falseproposal photos',len(hard),'top',[(r['id'],round(r['maximum_false_detection_score'],4)) for r in hard[:16]],flush=True)
