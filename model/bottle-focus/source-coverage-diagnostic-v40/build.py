"""Build a validation-only, source-reserved litter diagnostic before inference."""
from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import yaml

RUN=Path(__file__).resolve().parent
CACHE=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-source-diagnostic-v40')
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
sources=module('v40_sources',RUN/'compile_sources.py');reserve=module('v40_reserve',RUN/'reserve.py');scale=module('v40_scale',RUN/'scale.py')
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fingerprint(records):
 fields=['id','image_sha','pixels','source_group','rows','width','height']
 return hashlib.sha256(json.dumps([{k:r[k] for k in fields} for r in records],sort_keys=True).encode()).hexdigest()

def selection(compiled):
 records=compiled['records'];positive=[r for r in records if r['rows']];negative=[r for r in records if not r['rows']]
 eligible=[r for r in positive if scale.eligible(r)]
 old=compiled['permanent_monitor_records']
 chosen=reserve.split(eligible,negative,forbidden_ids={r['id'] for r in old},forbidden_shas={r['image_sha'] for r in old},forbidden_pixels={r['pixels'] for r in old},forbidden_groups={r['source_group'] for r in old})
 selected_ids={r['id'] for r in chosen['monitor']}
 chosen['remaining']=sorted([r for r in records if r['id'] not in selected_ids],key=lambda r:r['id'])
 chosen.update(all_positive_photos=len(positive),scale_eligible_positive_photos=len(eligible),all_negative_photos=len(negative),minimum_area_at640=scale.MIN_AREA_AT640)
 return chosen

def verify():
 audit=json.loads((RUN/'dataset.json').read_text());compiled=sources.compile_all();chosen=selection(compiled)
 assert audit['source_input_shas']==compiled['source_input_shas'] and audit['compiled_reviews']==compiled['compiled_reviews']
 assert audit['selection']==chosen and audit['cohort_fingerprint']==fingerprint(chosen['monitor'])
 assert audit['training_allowed'] is False and audit['original_benchmark_used'] is False and audit['independent_NHS_test'] is False
 expected=set()
 for source,row in zip(chosen['monitor'],audit['records'],strict=True):
  assert all(row[k]==v for k,v in source.items())
  assert sources.admission.hashes(row['image'])==(row['image_sha'],row['pixels'])
  assert Path(row['label']).read_text().splitlines()==row['rows']
  expected.add(Path(row['image']).name)
 assert {p.name for p in (CACHE/'images/val').iterdir()}==expected
 assert {p.name for p in (CACHE/'labels/val').glob('*.txt')}=={Path(n).stem+'.txt' for n in expected}
 assert yaml.safe_load((RUN/'data.yaml').read_text())==audit['config']
 assert audit['config']['train']=='/dev/null' and audit['photos']==32 and audit['positive_photos']==audit['negative_photos']==16
 assert audit['bottle_objects']==sum(len(r['rows']) for r in audit['records'])
 return audit

def main():
 compiled=sources.compile_all();chosen=selection(compiled);records=[]
 assert not (RUN/'measurement-start.json').exists(),'Source selection is locked after measurements start'
 for source in chosen['monitor']:
  image=CACHE/'images/val'/Path(source['source_image']).name;label=CACHE/'labels/val'/(image.stem+'.txt')
  image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
  if image.exists():assert sha(image)==source['image_sha']
  else:shutil.copyfile(source['source_image'],image)
  text='\n'.join(source['rows'])+('\n' if source['rows'] else '')
  if label.exists():assert label.read_text()==text
  else:label.write_text(text)
  records.append(dict(source,image=str(image),label=str(label)))
 config=dict(path=str(CACHE),train='/dev/null',val=str(CACHE/'images/val'),names=['bottle'])
 (RUN/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
 audit=dict(source_input_shas=compiled['source_input_shas'],compiled_reviews=compiled['compiled_reviews'],selection=chosen,records=records,
            cohort_fingerprint=fingerprint(chosen['monitor']),photos=32,positive_photos=16,negative_photos=16,bottle_objects=sum(len(r['rows']) for r in records),
            config=config,training_allowed=False,original_benchmark_used=False,independent_NHS_test=False,
            limitations='Fresh public-source development cohort with16size-eligible litter-positive and16independently AI-reviewed absence photos; medium/large mix still differs from benchmark; no causal source attribution or NHS independence. All selected groups and aliases remain excluded from future training.')
 (RUN/'dataset.json').write_text(json.dumps(audit,indent=2)+'\n');verify();print('LITTER_DIAGNOSTIC_FROZEN',audit['photos'],audit['bottle_objects'],audit['cohort_fingerprint'],flush=True)

if __name__=='__main__':main()
