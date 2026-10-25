"""Stream a bounded official label prefix for novel source candidates only."""
from pathlib import Path
import csv
import hashlib
import json
import subprocess

RUN=Path(__file__).resolve().parent
SOURCE='https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-human-imagelabels.csv'
CLASSES={'/m/02rlncx':'Plastic bottle','/m/0118n_9r':'Water bottle'}

if __name__=='__main__':
 prior_path=RUN.parent/'openimages-staging-v32/plastic-water-image-labels.json'
 monitor_path=RUN.parent/'monitor-campaign-v34/monitor-source-proposal.json'
 prior=json.loads(prior_path.read_text());monitor=json.loads(monitor_path.read_text())
 excluded={r['id'] for r in prior['records']}|{r['id'] for r in monitor['records']}
 process=subprocess.Popen(['curl','-fsSL','--max-time','900',SOURCE],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
 header=process.stdout.readline();checksum=hashlib.sha256(header);used=len(header);fields=next(csv.reader([header.decode()]));count=0
 current=None;positive=[];records=[];limit_reached=False
 def complete():
  if positive and current not in excluded:
   records.append(dict(id=current,positive_source_labels=list(positive)))
 try:
  for raw in process.stdout:
   row=dict(zip(fields,next(csv.reader([raw.decode()])),strict=True));identity=row['ImageID']
   assert current is None or identity>=current,'Official labels are not image-ID ordered'
   if identity!=current:
    if current is not None:complete()
    if len(records)>=1000 or used>=4*1024**3:
     limit_reached=True;break
    current=identity;positive=[]
   checksum.update(raw);used+=len(raw);count+=1
   if row['LabelName'] in CLASSES and float(row['Confidence'])==1:positive.append(row)
   if count%1000000==0:print('LABEL_PREFIX',count,used,'NOVEL_IDS',len(records),flush=True)
  else:
   assert process.wait()==0,'Incomplete official source download'
   if current is not None:complete()
 finally:
  process.stdout.close()
  if process.poll() is None:process.terminate()
  process.wait()
 assert records and not ({r['id'] for r in records}&excluded)
 result=dict(source_url=SOURCE,targets=CLASSES,source_bytes_processed=used,source_rows_processed=count,processed_prefix_sha256=checksum.hexdigest(),checksum_scope='Processed prefix through complete image groups, not full upstream file',records=records,excluded_prior_candidates_sha=hashlib.sha256(prior_path.read_bytes()).hexdigest(),excluded_monitor_proposal_sha=hashlib.sha256(monitor_path.read_bytes()).hexdigest(),training_admission=False,active_v36_changed=False,limitations='Image-level positives only; no source photo has been admitted. Complete object annotations, geometry, metadata, whole-photo AI reviews and overlap screening remain required.')
 (RUN/'novel-plastic-water-labels.json').write_text(json.dumps(result,indent=2)+'\n')
 print('NOVEL_SOURCE_FILTER_COMPLETE',len(records),used,count,flush=True)
