from pathlib import Path
from collections import Counter
import json
from PIL import Image,ImageOps,ImageDraw
RUN=Path('model/bottle-focus/optimization-diagnostic-v43');paths=list(Counter(Path('model/bottle-focus/reviewed-full-training-v42/training-exposures.txt').read_text().splitlines()))
records=[]
for name in paths:
 p=Path(name);label=p.parent.parent.parent/'labels'/p.parent.name/(p.stem+'.txt');rows=label.read_text().splitlines()
 if p.name.startswith('trash_detection__'):
  records.append(dict(image=str(p),rows=rows,name=p.name))
records.sort(key=lambda r:r['name']);(RUN/'trash-training-review-records.json').write_text(json.dumps(records,indent=2)+'\n')
for start in range(0,len(records),12):
 sheet=Image.new('RGB',(2048,1632),'white');draw=ImageDraw.Draw(sheet)
 for j,record in enumerate(records[start:start+12]):
  with Image.open(record['image']) as source:im=source.convert('RGB')
  d=ImageDraw.Draw(im);w,h=im.size
  for row in record['rows']:
   _,x,y,bw,bh=map(float,row.split());d.rectangle(((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h),outline='#00ff00',width=3)
  im=ImageOps.contain(im,(504,504));x=j%4*512;y=j//4*544;sheet.paste(im,(x,y));draw.text((x+3,y+507),str(start+j)+': '+record['name'][24:65],fill='black');draw.text((x+3,y+523),'GT boxes:'+str(len(record['rows'])),fill='black')
 sheet.save(RUN/f'trash-training-{start:03}.jpg',quality=95)
print('Review contact sheets',len(records))
