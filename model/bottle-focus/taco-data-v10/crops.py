from pathlib import Path
import json
from PIL import Image,ImageOps,ImageDraw
RUN=Path(__file__).resolve().parent
patches=[]
for r in json.loads((RUN/'taco-downloads.json').read_text()):
 if r['status']!='unreviewed':continue
 im=Image.open(r['image']).convert('RGB')
 for i,a in enumerate(r['annotations']):
  x,y,w,h=a['bbox'];pad=max(w,h)*.15;box=(max(0,int(x-pad)),max(0,int(y-pad)),min(im.width,int(x+w+pad)),min(im.height,int(y+h+pad)))
  patches.append((r['id'],i,im.crop(box),a['bbox']))
for start in range(0,len(patches),12):
 sheet=Image.new('RGB',(1536,1200),'white');d=ImageDraw.Draw(sheet)
 for j,(name,k,patch,box) in enumerate(patches[start:start+12]):
  patch=ImageOps.contain(patch,(370,350));x=j%4*384;y=j//4*400;sheet.paste(patch,(x,y));d.text((x+2,y+355),f'{name}: box {k} {box[2]:.0f}x{box[3]:.0f}',fill='black')
 sheet.save(RUN/f'crops-{start:03d}.jpg',quality=95)
