"""Reconnaissance: find human-positive plastic/water bottle image IDs, not training data."""
from pathlib import Path
import csv
import hashlib
import json
import subprocess

RUN=Path(__file__).resolve().parent
URL='https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-human-imagelabels.csv'
TARGETS={'/m/02rlncx':'Plastic bottle','/m/0118n_9r':'Water bottle'}
if __name__=='__main__':
    process=subprocess.Popen(['curl','-fsSL','--max-time','180',URL],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    checksum=hashlib.sha256();used=0;count=0;selected={}
    header=process.stdout.readline();checksum.update(header);used+=len(header);fields=next(csv.reader([header.decode()]))
    try:
        for raw in process.stdout:
            checksum.update(raw);used+=len(raw);count+=1
            row=dict(zip(fields,next(csv.reader([raw.decode()])),strict=True))
            if row['LabelName'] in TARGETS and float(row['Confidence'])==1:
                selected.setdefault(row['ImageID'],[]).append(row)
            if len(selected)>=500 or used>=1500*1024**2:break
        else:assert process.wait()==0,'Incomplete download'
    finally:
        process.stdout.close()
        if process.poll() is None:process.terminate()
        process.wait()
    assert selected
    report=dict(source_url=URL,class_source_url='https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv',targets=TARGETS,source_bytes_processed=used,source_rows_processed=count,processed_prefix_sha256=checksum.hexdigest(),checksum_scope='Consumed source prefix, not full upstream CSV',records=[dict(id=i,positive_source_labels=rows) for i,rows in selected.items()],training_admission=False,limitations='Image-level human-positive labels are only a candidate filter. They do not establish per-object bottle material, geometry, exhaustive annotation or eligibility.')
    (RUN/'plastic-water-image-labels.json').write_text(json.dumps(report,indent=2)+'\n');print('POSITIVE_PLASTIC_OR_WATER_CANDIDATE_IDS',len(selected),count,used,flush=True)
