"""Bounded data reconnaissance only: stage complete image groups, never train/admit."""
from pathlib import Path
import csv
import hashlib
import json
import subprocess

RUN=Path(__file__).resolve().parent
URL='https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv'
MID='/m/04dr76w'
if __name__=='__main__':
    process=subprocess.Popen(['curl','-fsSL','--max-time','120',URL],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    checksum=hashlib.sha256();processed_bytes=0;processed_rows=0;groups=[];current=None;rows=[]
    header=process.stdout.readline();checksum.update(header);processed_bytes+=len(header);fields=next(csv.reader([header.decode()]))
    def complete(image,boxes):
        if not boxes:return
        assert len({(b['XMin'],b['XMax'],b['YMin'],b['YMax']) for b in boxes})==len(boxes)
        for box in boxes:
            assert box['ImageID']==image and box['LabelName']==MID
            x,z,y,w=[float(box[k]) for k in ('XMin','XMax','YMin','YMax')]
            assert 0<=x<z<=1 and 0<=y<w<=1
        groups.append(dict(id=image,official_split='train',annotations=boxes,status='unreviewed_generic_bottle_not_admitted'))
    try:
        for raw in process.stdout:
            row=dict(zip(fields,next(csv.reader([raw.decode()])),strict=True));image=row['ImageID']
            assert current is None or image>=current,'Source image groups are not monotonic; cannot safely use prefix groups'
            if image!=current:
                if current is not None:complete(current,rows)
                if len(groups)>=100 or processed_bytes>=100*1024**2:break
                current=image;rows=[]
            checksum.update(raw);processed_bytes+=len(raw);processed_rows+=1
            if row['LabelName']==MID:rows.append(row)
        else:
            assert process.wait()==0,'Incomplete source download'
            if current is not None:complete(current,rows)
    finally:
        process.stdout.close()
        if process.poll() is None:process.terminate()
        process.wait()
    assert groups
    report=dict(source_url=URL,official_documentation='https://storage.googleapis.com/openimages/web/download_v7.html',boxable_class='/m/04dr76w Bottle',source_bytes_processed=processed_bytes,source_rows_processed=processed_rows,processed_prefix_sha256=checksum.hexdigest(),checksum_scope='Consumed prefix through last complete retained image group, not the entire upstream CSV',groups=groups,training_admission=False,limitations='Generic bottle scope includes glass/reusable containers. Images, source geometry/orientation, photo licensing, exhaustive target annotation, fixed-validation overlap and independent visual review must all be checked before any future training admission.')
    (RUN/'annotation-prefix.json').write_text(json.dumps(report,indent=2)+'\n');print('OPENIMAGES_GENERIC_CANDIDATE_GROUPS',len(groups),processed_rows,processed_bytes,flush=True)
