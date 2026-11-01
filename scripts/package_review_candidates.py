"""Freeze a provisional visual-review bundle; never marks unresolved photos independent."""
from pathlib import Path
import json,csv,shutil,zipfile,hashlib,textwrap
from collections import Counter
from datetime import datetime,timezone
import sys
from PIL import Image,ImageOps,ImageDraw,ImageFont
sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_audit as a
import specialists as s
ROOT=s.ROOT.parent/(s.ROOT.name+'-assets')/'independent-evaluation-20261101'
freeze=s.read(s.ROOT/'configs/research-freeze.json');a.verify_freeze(freeze)
candidates=s.read(ROOT/'sources/candidates.json')
manual=[json.loads(line) for line in (ROOT/'individual-review.jsonl').read_text().splitlines()]
assert len(manual)==100 and len({r['review_id'] for r in manual})==100
reviews={r['review_id']:r for r in manual};overlap={r['review_id']:r for r in s.read(ROOT/'sources/overlap-report.json')['candidates']}
phash={r['review_id']:r for r in s.read(ROOT/'sources/phash-report.json')['candidates']}
provenance=s.read(ROOT/'sources/provenance-assessment.json');pairs={r['review_id']:r['pair'] for r in provenance['source_pairs']}
# All 100 were visually inspected before this conservative redundancy screen.
# Global hashes are review flags; these removals do not assert the images are duplicates.
remove={'LF033':['LF023','LF025'],'LF035':['LF002'],'LF040':['LF014'],'LF059':['LF006'],'LF067':['LF002','LF035','LF036'],
        'LF069':['LF006','LF014'],'LF090':['LF006','LF056','LF069'],'LF095':['LF005'],'LF096':['LF002','LF067']}
package=ROOT/'evaluation-candidates';images=package/'images'
assert not package.exists(),'Refusing to overwrite an existing assembled cohort'
images.mkdir(parents=True)
rows=[];selected=[]
for c in candidates:
 rid=c['review_id'];r=reviews[rid];o=overlap[rid];p=phash[rid]
 assert s.sha(c['original'])==c['source_sha256']
 assert not o['historical_exact_byte_paths'] and not o['historical_exact_pixel_shas']
 status=r['visual_status'];reason=r['visual_reason']
 if rid in remove:
  assert status=='KEEP';status='REJECT'
  reason+=' Conservative redundancy exclusion: similar global food/plate layout to '+', '.join(remove[rid])+'. Different source pair IDs; this is not proof of exact duplication.'
 export_name=rid+'__'+Path(c['filename']).stem+'.jpg'
 if status=='KEEP':
  shutil.copyfile(c['original'],images/export_name)
  assert s.sha(images/export_name)==c['source_sha256']
  selected.append(dict(review_id=rid,filename=export_name,original_filename=c['filename'],sha256=c['source_sha256'],source_pair_id=pairs[rid]['source_pair_id'],
                       before_pair_filename=pairs[rid]['before_filename'],source_url=c['source_url'],license='CC BY 4.0',independence_status='UNRESOLVED',independent_eligible=False))
 histogram_flags=[]
 if o['historical_near_match']:histogram_flags.append('dHash flag visually cleared: historical image has a different subject/scene')
 if p['historical_phash_match']:histogram_flags.append('pHash flag visually cleared: comparator is juice carton, candidate is food plate')
 flags='; '.join(histogram_flags) or 'No historical perceptual flag at declared radii'
 row=dict(review_id=rid,filename=c['filename'],export_filename=export_name if status=='KEEP' else '',source='LeFood-Set V1; Sari, Wani, Nakazawa (2025)',
   source_doi='10.17632/cchsk79jkt.1',source_url=c['source_url'],license='CC BY 4.0',width=c['width'],height=c['height'],
   relevant_classes=r['relevant_classes'],review_status=status,initial_visual_status=r['visual_status'],reason=reason,
   cafeteria_relevance='Controlled Indonesian plate-leftover imagery; not mixed school waste or a full seven-class benchmark',
   food_presence='UNCERTAIN' if 'food?' in r['relevant_classes'] else ('YES' if 'food' in r['relevant_classes'] else 'NO'),plate_presence='YES',
   utensil_presence='NO',hinged_container_presence='NO',drink_carton_presence='NO',wrapper_presence='NO',bottle_presence='NO',
   annotation_readiness='Candidate for human annotation; no boxes or bin truth assigned' if status=='KEEP' else 'Not selected; see reason',
   disposal_ambiguity='Plate reuse/material and actual bin eligibility not established; no human disposal labels',
   source_pair_id=pairs[rid]['source_pair_id'],source_before_image=pairs[rid]['before_filename'],capture_datetime='UNAVAILABLE',
   historical_byte_overlap='NONE_FOUND',historical_pixel_overlap='NONE_FOUND',historical_perceptual_findings=flags,
   within_candidate_near_flags=json.dumps(o['within_candidate_flags'],separators=(',',':')),
   provenance_status='UNRESOLVED',independently_verified='NO',
   provenance_reason='No known overlap against available historical files, but complete source/capture lineage and historical reuse cannot be ruled out; newly downloaded does not imply independence',
   human_verified='NO',reviewer='Assistant individual visual inspection',sha256=c['source_sha256'])
 rows.append(row)
assert len(selected)==52
assert len({r['before_pair_filename'] for r in selected})==52
# Do not include any pair flagged as near-similar among the exported 52.
selected_ids={r['review_id'] for r in selected}
assert not [(r['review_id'],f['review_id']) for r in rows if r['review_id'] in selected_ids for f in overlap[r['review_id']]['within_candidate_flags'] if f['review_id'] in selected_ids]
report=ROOT/'individual-review.csv'
with report.open('w',newline='',encoding='utf-8-sig') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
shutil.copyfile(report,package/report.name)
font='/System/Library/Fonts/Supplemental/Arial.ttf'
try:label_font=ImageFont.truetype(font,15);title_font=ImageFont.truetype(font,22)
except OSError:label_font=title_font=ImageFont.load_default()

def sheet(items,output,columns,title):
 cellw,cellh=224,198;header=85;countrows=(len(items)+columns-1)//columns
 canvas=Image.new('RGB',(columns*cellw,header+countrows*cellh),'white');draw=ImageDraw.Draw(canvas)
 draw.text((15,12),title,font=title_font,fill='black')
 draw.text((15,45),'AI visual review only. All source provenance UNRESOLVED. No detector inference or ground-truth labels.',font=label_font,fill='#8d3300')
 for index,r in enumerate(items):
  x=(index%columns)*cellw;y=header+(index//columns)*cellh
  c=next(c for c in candidates if c['review_id']==r['review_id'])
  with Image.open(c['original']) as original:im=ImageOps.exif_transpose(original).convert('RGB');im.thumbnail((cellw-8,cellh-38))
  canvas.paste(im,(x+(cellw-im.width)//2,y))
  color={'KEEP':'#146134','REJECT':'#a02020','UNCERTAIN':'#9a6500'}[r['review_status']]
  draw.text((x+5,y+cellh-33),r['review_id']+' '+r['review_status'],font=label_font,fill=color)
 canvas.save(output,quality=93)
sheet([r for r in rows if r['review_status']=='KEEP'],package/'contact-sheet-selected.jpg',7,'52 provisional visual KEEP candidates — NOT a verified independent test')
sheet(rows,ROOT/'contact-sheet-all100.jpg',8,'All 100 individually inspected candidates: 52 KEEP / 35 REJECT / 13 UNCERTAIN')
cohort=dict(status='PROVISIONAL_REVIEW_COHORT_NOT_INDEPENDENTLY_CERTIFIED',selection_frozen=True,selection_rule='Individual visual review, then conservative radius5 dHash within-cohort redundancy exclusion; no model outputs used',
 annotation_rules_frozen=False,inference_authorized=False,independently_verified_images=0,
 annotation_policy_pending=['Human confirm food-remains versus bone/cartilage/sauce-only labels and minimum visible fragments','Human confirm one-box-per-serving grouping and visible-extent boxes on partly clipped plates','Human review proposed class presence; no source weights/ingredient metadata are disposal labels'],
 source='https://data.mendeley.com/datasets/cchsk79jkt/1',doi='10.17632/cchsk79jkt.1',license='CC BY 4.0',license_url='https://creativecommons.org/licenses/by/4.0/',
 attribution='Sari, Yuita Arum; Wani, Yudi Arimba; Nakazawa, Atsushi (2025), LeFood-Set: Leftovers Food Dataset, Mendeley Data, V1, doi:10.17632/cchsk79jkt.1',
 original_pixels_modified=False,renaming_only=True,images=selected)
(package/'cohort-manifest.json').write_text(json.dumps(cohort,indent=2)+'\n')
(package/'ATTRIBUTION.txt').write_text(cohort['attribution']+'\nSource: '+cohort['source']+'\nLicense: CC BY 4.0 — '+cohort['license_url']+'\n52 source JPEGs copied without pixel changes; filenames prefixed with review IDs. Contact sheet is resized for review.\nOriginal author/source license applies; no author endorsement implied.\n')
(package/'README.txt').write_text('''PROVISIONAL REVIEW CANDIDATES — NOT A VERIFIED INDEPENDENT EVALUATION SET

52 photographs were individually visually accepted after reviewing 100 candidates.
Final visual disposition: 52 KEEP, 35 REJECT, 13 UNCERTAIN. Nine initial visual KEEP
images were excluded conservatively for near-similar layout/content.
All 52 exported files have UNRESOLVED historical source/capture provenance.
Independently verified images: ZERO. Do not report these as independent test accuracy.
The supplied CSV covers every reviewed candidate and separates visual eligibility
from provenance. There are no human-certified class or disposal labels.

Coverage: food and plates only. No visible bottle, carton, wrapper, utensil, or
hinged-container examples were found. This is NOT a seven-class benchmark.
Background/photography are controlled plate-leftover photography, not a new
capture session at the user's school. Plate rims are sometimes clipped; food
servings in selected cases are visible. Bone/sauce/fragment policies need human
confirmation before any boxes are assigned. No bin labels or detector outputs
are included. Model inference is not authorized until selection AND annotation
rules are frozen and provenance eligibility is resolved.

Historical comparison: 103861 readable historical image paths plus eight relevant
Downloads dataset ZIP archives (1961 image members) and six project ZIPs
(87 image members; two contain only annotations), representing 14698 byte-unique
historical images. Exact bytes/pixels: zero candidate matches. Two classical
perceptual screens and individual comparator review found false-similarity flags,
not matching photos. There are 42 dangling/unreadable historical aliases and
incomplete earlier metadata; absence of a duplicate is not proof of independence.
100 source before/after pairs were matched in the publisher's workbook. Images
lack capture-time EXIF, and complete historical source reuse is unverified.

Source and CC BY4.0 attribution are in ATTRIBUTION.txt. Original JPEG bytes are
preserved in images/. Contact sheets are review aids, not original photographs.
''')
zip_path=ROOT/'evaluation-candidates-52.zip'
assert not zip_path.exists(),'Refusing to overwrite existing ZIP'
with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for path in sorted(package.rglob('*')):
  if path.is_file():z.write(path,path.relative_to(package).as_posix())
with zipfile.ZipFile(zip_path) as z:
 assert z.testzip() is None
 members=z.namelist();imnames=[n for n in members if n.startswith('images/')]
 assert len(imnames)==52 and len(set(members))==len(members)
 assert all(not Path(n).is_absolute() and '..' not in Path(n).parts and '__MACOSX' not in n for n in members)
 for row in selected:assert hashlib.sha256(z.read('images/'+row['filename'])).hexdigest()==row['sha256']
 # Validate the exact extracted image set, including spaces in the extraction path.
 import tempfile
 with tempfile.TemporaryDirectory(prefix='cafeteria upload verify ') as tmp:
  z.extractall(tmp)
  for row in selected:assert s.sha(Path(tmp)/'images'/row['filename'])==row['sha256']
a.verify_freeze(freeze)
summary=dict(individually_inspected_candidates=100,historical_reference_images_visually_inspected=3,
 final_visual_status_counts=dict(Counter(r['review_status'] for r in rows)),initial_visual_status_counts=dict(Counter(r['visual_status'] for r in manual)),
 selected_provisional_images=52,independently_verified_images=0,provenance_unresolved_candidates=100,
 class_presence_in_selected=dict(food=51,plate=52,utensil=0,hinged_container=0,drink_carton=0,wrapper=0,bottle=0),
 no_inference=True,no_training=True,active_freeze_verified=True,ground_truth_labels_generated=False,
 folder=str(package.resolve()),csv=str(report.resolve()),contact_sheet=str((package/'contact-sheet-selected.jpg').resolve()),all_review_contact_sheet=str((ROOT/'contact-sheet-all100.jpg').resolve()),zip=str(zip_path.resolve()),
 zip_images=52,zip_entries=len(members),zip_bytes=zip_path.stat().st_size,zip_sha256=s.sha(zip_path),zip_crc_and_roundtrip_extraction_passed=True,
 limitations=['The requested 50–100 independently certified images were not established; 52 is a provisional visual-review bundle only.','All source/capture provenance remains unresolved despite no known historical pixel/byte overlap.','Five target classes are absent; no claim of complete model/class coverage is possible.'],
 diagnostics=['Legacy repo data/ and src/prepare.py and local .venv paths are absent; external asset paths and external Python used.',
              'Exact previous ZIP commands were not in available history; no specific original shell error is claimed.',
              'Mendeley guessed versions/... endpoints returned404 and root listing was empty; fixed by published public folder tree/file APIs and verified publisher SHA256.',
              'SciPy was absent; classical pHash was implemented using equivalent NumPy orthonormal DCT, without changing the existing environment.',
              'ZIP created with pathlib/zipfile, explicit relative members, no shell globs, no resource forks; CRC and extraction/hash tests passed.'])
assert summary['class_presence_in_selected']['food']==sum(row['food_presence']=='YES' for row in rows if row['review_status']=='KEEP')
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
