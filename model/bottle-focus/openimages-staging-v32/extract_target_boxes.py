"""Stage complete official bottle boxes for targeted human-positive image IDs."""
from pathlib import Path
import csv
import hashlib
import json
import subprocess

RUN=Path(__file__).resolve().parent
URL='https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv'
if __name__=='__main__':
    labels=json.loads((RUN/'plastic-water-image-labels.json').read_text());wanted={r['id']:r for r in labels['records']};maximum=max(wanted)
    process=subprocess.Popen(['curl','-fsSL','--max-time','180',URL],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    checksum=hashlib.sha256();used=0;count=0;groups=[];current=None;rows=[]
    header=process.stdout.readline();checksum.update(header);used+=len(header);fields=next(csv.reader([header.decode()]))
    def complete(image,boxes):
        if not boxes:return
        for box in boxes:
            assert box['ImageID']==image and box['LabelName']=='/m/04dr76w'
            x,z,y,w=[float(box[k]) for k in ('XMin','XMax','YMin','YMax')]
            assert 0<=x<z<=1 and 0<=y<w<=1
        groups.append(dict(id=image,official_split='train',annotations=boxes,positive_source_labels=wanted[image]['positive_source_labels'],status='unreviewed_targeted_candidate_not_admitted'))
    try:
        for raw in process.stdout:
            row=dict(zip(fields,next(csv.reader([raw.decode()])),strict=True));image=row['ImageID'];assert current is None or image>=current
            if image!=current:
                if current is not None:complete(current,rows)
                if image>maximum:break
                current=image;rows=[]
            checksum.update(raw);used+=len(raw);count+=1
            if image in wanted and row['LabelName']=='/m/04dr76w':rows.append(row)
        else:
            assert process.wait()==0,'Incomplete source download'
            if current is not None:complete(current,rows)
    finally:
        process.stdout.close()
        if process.poll() is None:process.terminate()
        process.wait()
    report=dict(source_url=URL,image_label_filter_sha256=hashlib.sha256((RUN/'plastic-water-image-labels.json').read_bytes()).hexdigest(),source_bytes_processed=used,source_rows_processed=count,processed_prefix_sha256=checksum.hexdigest(),checksum_scope='Consumed prefix through complete source image groups; not full upstream CSV',groups=groups,positive_image_ids_without_bottle_boxes=sorted(set(wanted)-{r['id'] for r in groups}),training_admission=False,limitations='Candidate filter only; each object still requires scope/box review, photo metadata, actual-image geometry, frozen-validation overlap screening and independent QA before any future training admission.')
    (RUN/'targeted-annotation-prefix.json').write_text(json.dumps(report,indent=2)+'\n');print('TARGETED_BOXED_IMAGES',len(groups),'BOXES',sum(len(r['annotations']) for r in groups),'ROWS',count,flush=True)
