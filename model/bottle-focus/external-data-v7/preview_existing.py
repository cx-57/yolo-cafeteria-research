"""Sample existing training closeups for a label-definition quality check."""
from pathlib import Path
import json,random
from PIL import Image,ImageDraw,ImageOps
RUN=Path(__file__).resolve().parent
DATA=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-capture-validation')
m=json.loads((DATA/'manifest.json').read_text());samples=[]
for prefix in ['dry_wet_waste','waste_detection_v3']:
 r=[r for r in m['records'] if r['split']=='train' and r['rows'] and r['name'].startswith(prefix)]
 samples.extend(random.Random(42).sample(r,12))
(RUN/'existing-sample.json').write_text(json.dumps(samples,indent=2)+'\n')
for start in range(0,len(samples),12):
 sheet=Image.new('RGB',(1536,1400),'white');d=ImageDraw.Draw(sheet)
 for j,r in enumerate(samples[start:start+12]):
  im=Image.open(DATA/'images/train'/r['name']).convert('RGB');im=ImageOps.contain(im,(500,305));x=j%3*512;y=j//3*350;sheet.paste(im,(x,y));d.text((x+2,y+308),f"{start+j}: {r['name'].split('__')[-1][:45]}",fill='black')
 sheet.save(RUN/f'existing-preview-{start:03d}.jpg',quality=90)
