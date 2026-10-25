"""Stage complete official bottle groups and photo metadata for novel candidates."""
from pathlib import Path
import csv
import hashlib
import json
import subprocess

RUN=Path(__file__).resolve().parent
BBOX='https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv'
METADATA='https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv'

def boxes():
 label_path=RUN/'novel-plastic-water-labels.json';labels=json.loads(label_path.read_text())
 wanted={r['id']:r for r in labels['records']};maximum=max(wanted)
 process=subprocess.Popen(['curl','-fsSL','--max-time','600',BBOX],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
 header=process.stdout.readline();checksum=hashlib.sha256(header);used=len(header);fields=next(csv.reader([header.decode()]));count=0;groups=[];current=None;annotations=[]
 def complete():
  if annotations:
   groups.append(dict(id=current,official_split='train',annotations=list(annotations),positive_source_labels=wanted[current]['positive_source_labels'],status='unreviewed_staged_only'))
 try:
  for raw in process.stdout:
   row=dict(zip(fields,next(csv.reader([raw.decode()])),strict=True));identity=row['ImageID'];assert current is None or identity>=current
   if identity!=current:
    if current is not None:complete()
    if identity>maximum:break
    current=identity;annotations=[]
   checksum.update(raw);used+=len(raw);count+=1
   if identity in wanted and row['LabelName']=='/m/04dr76w':
    x,z,y,t=[float(row[k]) for k in ['XMin','XMax','YMin','YMax']];assert 0<=x<z<=1 and 0<=y<t<=1;annotations.append(row)
   if count%500000==0:print('BBOX_PREFIX',count,used,'GROUPS',len(groups),flush=True)
  else:
   assert process.wait()==0,'Incomplete source bbox download'
   if current is not None:complete()
 finally:
  process.stdout.close()
  if process.poll() is None:process.terminate()
  process.wait()
 report=dict(source_url=BBOX,image_label_filter_sha256=hashlib.sha256(label_path.read_bytes()).hexdigest(),source_bytes_processed=used,source_rows_processed=count,processed_prefix_sha256=checksum.hexdigest(),checksum_scope='Processed prefix through complete source image groups, not full upstream file',groups=groups,positive_ids_without_generic_bottle_boxes=sorted(set(wanted)-{r['id'] for r in groups}),training_admission=False,active_v36_changed=False)
 (RUN/'novel-complete-boxes.json').write_text(json.dumps(report,indent=2)+'\n');print('COMPLETE_SOURCE_GROUPS',len(groups),sum(len(r['annotations']) for r in groups),flush=True)
 return groups

def metadata(groups):
 wanted={r['id'] for r in groups};found=[];checksum=hashlib.sha256();used=0;count=0
 process=subprocess.Popen(['curl','-fsSL','--max-time','600',METADATA],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
 def lines():
  nonlocal used
  for raw in process.stdout:
   checksum.update(raw);used+=len(raw);yield raw.decode('utf8')
 try:
  for row in csv.DictReader(lines()):
   count+=1
   if row['ImageID'] in wanted:found.append(row)
   if count%1000000==0:print('METADATA_ROWS',count,'FOUND',len(found),flush=True)
  assert process.wait()==0,'Incomplete photo metadata download'
 finally:
  process.stdout.close()
  if process.poll() is None:process.terminate()
  process.wait()
 assert len({r['ImageID'] for r in found})==len(found)
 report=dict(source_url=METADATA,processed_rows=count,processed_bytes=used,source_sha256=checksum.hexdigest(),checksum_scope='Full upstream photo-metadata CSV as retrieved',found=found,missing=sorted(wanted-{r['ImageID'] for r in found}),training_admission=False,active_v36_changed=False)
 (RUN/'novel-photo-metadata.json').write_text(json.dumps(report,indent=2)+'\n');print('PHOTO_METADATA_COMPLETE',len(found),'MISSING',len(report['missing']),flush=True)

if __name__=='__main__':
 metadata(boxes())
