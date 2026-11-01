"""Package the 44 GitHub-source images with editable, explicitly unverified boxes.

Run ONLY after package_screened44.py rebuilds the exact historical 44 photos.
No source photo or annotation is promoted to independent test ground truth.
"""
import argparse
import hashlib
import json
from pathlib import Path

NAMES=['food','plate','utensil','hinged_container','drink_carton','wrapper','bottle']
TEMPLATE="""<!doctype html>
<meta charset="utf-8">
<title>44-photo seven-class annotation review</title>
<style>
*{box-sizing:border-box}body{font:15px system-ui;margin:0;color:#223;background:#f1f5f9}
header{background:white;padding:18px 24px}main{max-width:1500px;margin:20px auto;padding:0 18px}
h1{margin:0 0 8px;font-size:24px}h3{margin:10px 0 6px}p{margin:8px 0}
.warn{background:#fff2c8;border-left:5px solid #b8860b;padding:12px 14px;margin:10px 0}
.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:12px 0}
button,select,input{padding:6px;border:1px solid #aab7c7;border-radius:4px;background:white}
button{cursor:pointer}button.good{background:#174bbb;color:white}
.layout{display:grid;grid-template-columns:minmax(370px,1fr) 350px;gap:14px}
article,aside{border:1px solid #cbd5e1;background:white;padding:12px;border-radius:7px}
canvas{width:100%;height:auto;border:1px solid #cbd5e1;touch-action:none;cursor:crosshair}
#list{max-height:320px;overflow:auto;border:1px solid #ccd}
#list button{display:block;width:100%;text-align:left;border-radius:0;padding:8px}
#list button.selected{background:#dcecff}
#meta{color:#556;font-size:13px;overflow-wrap:anywhere}
#notes{width:100%;height:70px}label{display:block;margin:8px 0}
@media(max-width:920px){.layout{grid-template-columns:1fr}}
</style>
<header><h1>44-image annotation review (V3 vs original)</h1>
<p>104 AI-draft boxes on the original 44 public-source photographs. Every box is <b>unverified</b>.</p>
<div class="warn">These images were already inspected during model development, so this is <b>not an independent test</b>. Verify every visible object of all seven classes; source hints do not establish ground-truth labels. Images 34–36 contain crowded bottle scenes and are excluded by default in scoring until exhaustively labeled.</div></header>
<main>
<div class="controls">
<button id="prev">Previous</button><select id="imageChoice"></select><button id="next">Next</button>
<span id="progress"></span><button id="export" class="good">Export review JSON</button>
<label>Resume JSON <input id="import" type="file" accept=".json"></label>
</div>
<div class="layout"><article>
<p id="meta"></p><canvas id="photo" width="1000" height="750"></canvas>
<p>Drag to add a box. To fix a draft, delete it and draw a corrected box. Red = unverified; green = confirmed.</p>
<div class="controls"><label>New class <select id="className"></select></label><button id="draw">Draw boxes</button><button id="select">Select boxes</button></div>
<div class="warn" id="attention"></div>
</article><aside>
<h3>Boxes in this photo</h3><div id="list"></div>
<div class="controls"><button id="confirm">Confirm selected box</button><button id="remove">Delete selected box</button></div>
<label><input type="checkbox" id="reviewed"> I inspected this photo and confirmed all visible relevant objects and every box</label>
<label>Review notes<textarea id="notes"></textarea></label>
<p>Download the export after each session. You can re-import it to resume. The final evaluation refuses any unreviewed photo or unconfirmed box.</p>
</aside></div></main>
<script>
const DATA=__DATA__;
const NAMES=__NAMES__;
const records=DATA.images.map(x=>({id:x.id,path:x.path,sha256:x.sha256,source:x.source,
  reviewed:false,notes:'',boxes:x.draft_boxes.map(b=>({...b}))}));
let at=0,selected=-1,mode='select',img=null,drag=null;
const $=id=>document.getElementById(id),cv=$('photo'),ctx=cv.getContext('2d');
const clamp=x=>Math.max(0,Math.min(1,x));
NAMES.forEach(name=>{let x=document.createElement('option');x.textContent=name;$('className').append(x)});
DATA.images.forEach((x,i)=>{let o=document.createElement('option');o.value=i;o.textContent=String(i+1).padStart(2,'0')+' '+x.path.split('/').pop();$('imageChoice').append(o)});
function progress(){let n=records.filter(x=>x.reviewed).length;$('progress').textContent='Reviewed '+n+'/44; boxes '+records.reduce((a,x)=>a+x.boxes.length,0);}
function imageLoad(){let r=records[at];img=new Image();img.onload=()=>{cv.width=img.naturalWidth;cv.height=img.naturalHeight;draw()};img.onerror=()=>{ctx.clearRect(0,0,cv.width,cv.height);alert('Cannot load image '+r.path)};img.src=r.path}
function draw(){
 if(!img||!img.complete||!img.naturalWidth)return;
 ctx.drawImage(img,0,0,cv.width,cv.height);let r=records[at];
 r.boxes.forEach((b,i)=>{let [x1,y1,x2,y2]=b.box_xyxy_normalized;
 ctx.lineWidth=Math.max(3,cv.width/400);ctx.strokeStyle=b.status==='HUMAN_REVIEWED'?'#16a34a':'#e11d48';
 if(i===selected)ctx.strokeStyle='#2563eb';
 ctx.strokeRect(x1*cv.width,y1*cv.height,(x2-x1)*cv.width,(y2-y1)*cv.height);
 ctx.font=Math.max(13,cv.width/60)+'px system-ui';ctx.fillStyle=ctx.strokeStyle;
 ctx.fillText(b.class_name,(x1*cv.width)+2,Math.max(16,y1*cv.height-5));});
}
function showBoxes(){const r=records[at];$('list').replaceChildren();
 r.boxes.forEach((b,i)=>{let e=document.createElement('button');e.textContent=(i+1)+'. '+b.class_name+' ['+b.box_xyxy_normalized.map(x=>x.toFixed(2)).join(',')+'] '+(b.status==='HUMAN_REVIEWED'?'✔ CONFIRMED':'⚠ DRAFT');
 e.className=i===selected?'selected':'';e.onclick=()=>{selected=i;showBoxes();draw()};$('list').append(e)});
 $('reviewed').checked=r.reviewed;$('notes').value=r.notes||'';progress();}
function load(n){at=(n+44)%44;selected=-1;let m=DATA.images[at];$('imageChoice').value=at;
 $('meta').textContent='Image '+(at+1)+'/44 — '+m.source_hint+' (SOURCE HINT ONLY). SHA256 '+m.sha256;
 $('attention').textContent=m.scoring_eligible?'Eligible after complete manual review':'Previously excluded from default scoring (crowded/ambiguous). Thoroughly review before explicitly including.';
 imageLoad();showBoxes()}
function xy(ev){let rect=cv.getBoundingClientRect();return [clamp((ev.clientX-rect.left)/rect.width),clamp((ev.clientY-rect.top)/rect.height)]}
cv.onpointerdown=ev=>{if(mode==='draw'){drag=xy(ev);cv.setPointerCapture(ev.pointerId)}else{
 let [x,y]=xy(ev);selected=records[at].boxes.findIndex(b=>{let v=b.box_xyxy_normalized;return v[0]<=x&&x<=v[2]&&v[1]<=y&&y<=v[3]});showBoxes();draw()}};
cv.onpointerup=ev=>{if(!drag)return;let end=xy(ev),box=[Math.min(end[0],drag[0]),Math.min(end[1],drag[1]),Math.max(end[0],drag[0]),Math.max(end[1],drag[1])];drag=null;
 if(box[2]-box[0]<.005||box[3]-box[1]<.005)return;
 records[at].boxes.push({class_name:$('className').value,box_xyxy_normalized:box,bin:'unknown',evidence:'',status:'AI_DRAFT_UNVERIFIED'});records[at].reviewed=false;selected=records[at].boxes.length-1;showBoxes();draw()};
$('confirm').onclick=()=>{let b=records[at].boxes[selected];if(!b)return alert('Select a box first');b.status='HUMAN_REVIEWED';showBoxes();draw()};
$('remove').onclick=()=>{if(selected<0)return;records[at].boxes.splice(selected,1);records[at].reviewed=false;selected=-1;showBoxes();draw()};
$('reviewed').onchange=ev=>{let r=records[at];if(ev.target.checked&&r.boxes.some(b=>b.status!=='HUMAN_REVIEWED')){ev.target.checked=false;return alert('Confirm every remaining box before reviewing image')};r.reviewed=ev.target.checked;progress()};
$('notes').oninput=ev=>records[at].notes=ev.target.value;
$('next').onclick=()=>load(at+1);$('prev').onclick=()=>load(at-1);$('imageChoice').onchange=ev=>load(Number(ev.target.value));
$('draw').onclick=()=>mode='draw';$('select').onclick=()=>mode='select';
$('export').onclick=()=>{let payload={format:'CAFETERIA_PROVISIONAL_HUMAN_REVIEW_V1',not_independent:true,
 reviewed_image_count:records.filter(x=>x.reviewed).length,
 warning:'User-entered review state; source images were development-exposed and cannot form an untouched independent test.',images:records};
 let blob=new Blob([JSON.stringify(payload,null,2)+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');
 a.href=url;a.download='cafeteria-review-export.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
$('import').onchange=async ev=>{let f=ev.target.files[0];if(!f)return;try{
 let imported=JSON.parse(await f.text());if(imported.format!=='CAFETERIA_PROVISIONAL_HUMAN_REVIEW_V1'||imported.images?.length!==44)throw Error('Not a 44 photo review JSON');
 for(let i=0;i<44;i++){let a=records[i],b=imported.images[i];if(a.id!==b.id||a.path!==b.path||a.sha256!==b.sha256||!Array.isArray(b.boxes))throw Error('Image identity changed at '+i);
 for(let item of b.boxes){let v=item.box_xyxy_normalized;if(!NAMES.includes(item.class_name)||!Array.isArray(v)||v.length!==4||!v.every(Number.isFinite)||v[0]<0||v[1]<0||v[2]>1||v[3]>1||v[0]>=v[2]||v[1]>=v[3])throw Error('Invalid box at '+i)}}
 imported.images.forEach((x,i)=>records[i]=x);load(at);alert('Review loaded; export again before leaving.')
 }catch(err){alert('Cannot import: '+err.message)}ev.target.value=''};
load(0);
</script>"""

def assemble(folder,coords):
    folder=Path(folder)
    manifest_path=folder/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    images=manifest.get('images',[])
    if len(images)!=44 or len(coords)!=44:
        raise ValueError('Expected exactly 44 source images and 44 sets of drafts')
    all_boxes=0
    for i,(r,spec) in enumerate(zip(images,coords)):
        if spec[0]!=i or not isinstance(spec[1],list):
            raise ValueError('Draft indexing mismatch')
        image=folder/r['path']
        if not image.is_file() or hashlib.sha256(image.read_bytes()).hexdigest()!=r['sha256']:
            raise ValueError('Source SHA mismatch: '+str(image))
        r['id']=Path(r['path']).name
        r['scoring_eligible']=bool(r.get('scoring_eligible',True))
        r['draft_boxes']=[]
        for cls,box in spec[1]:
            if cls not in NAMES or len(box)!=4 or not all(0<=n<=1 for n in box) or not (box[0]<box[2] and box[1]<box[3]):
                raise ValueError(f'Bad draft box in {i+1}')
            r['draft_boxes'].append(dict(class_name=cls,box_xyxy_normalized=box,
                    bin='unknown',evidence='',status='AI_DRAFT_UNVERIFIED'))
            all_boxes+=1
    if all_boxes!=104:
        raise ValueError('Expected 104 labeled AI drafts')
    manifest.update(classes=NAMES,expected_test_images=44, human_verified_images=0,
        human_verified_detection_labels=0,
        independent_certified_images=0,data_role='DEVELOPMENT_DIAGNOSTIC_NOT_INDEPENDENT',
        status='AI_DRAFTS_NEED_MANUAL_OBJECT_ANNOTATION')
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    data_json=json.dumps(manifest,separators=(',',':')).replace('</','<\\/')
    (folder/'index.html').write_text(TEMPLATE.replace('__DATA__',data_json).
                                     replace('__NAMES__',json.dumps(NAMES)))
    rows=[dict(id=r['id'],path=r['path'],sha256=r['sha256'],
              source=r['source'],reviewed=False,notes='',boxes=r['draft_boxes']) for r in images]
    (folder/'cafeteria-review-export.json').write_text(json.dumps(dict(
        format='CAFETERIA_PROVISIONAL_HUMAN_REVIEW_V1',not_independent=True,
        reviewed_image_count=0,warning='UNREVIEWED draft JSON: use index.html before scoring.',
        images=rows),indent=2)+'\n')
    (folder/'README.txt').write_text(
        '44 publicly sourced photos with 104 unverified AI-draft boxes.\\n'
        'Open index.html after extracting this artifact. Confirm each box and each photo; export your own updated review JSON.\\n'
        'The included cafeteria-review-export.json is an UNREVIEWED STARTER: the evaluator must reject it until checked.\\n'
        'Photos 34-36 contain dense bottle scenes; excluded from default AP without exhaustive annotation.\\n'
        'This 44-photo cohort is development-exposed and does NOT support independent test accuracy claims.\\n')
    print(json.dumps(dict(images=len(images),AI_draft_boxes=all_boxes,verified=0,folder=str(folder))))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--folder',required=True,type=Path)
    ap.add_argument('--drafts',default='configs/review44-ai-draft-boxes.json',type=Path)
    args=ap.parse_args()
    assemble(args.folder,json.loads(args.drafts.read_text()))

if __name__=='__main__':main()
