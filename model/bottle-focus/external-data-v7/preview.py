"""Render full-image, annotation-numbered sheets for candidate review."""
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageOps
RUN = Path(__file__).resolve().parent
records = json.loads((RUN / 'downloads.json').read_text())
for start in range(0,len(records),6):
    sheet = Image.new('RGB', (1536, 1000), 'white')
    draw = ImageDraw.Draw(sheet)
    for j, r in enumerate(records[start:start+6]):
        im = ImageOps.exif_transpose(Image.open(r['image'])).convert('RGB')
        w,h = im.size
        annotations = []
        for k,line in enumerate(Path(r['label']).read_text().splitlines()):
            c,x,y,bw,bh = map(float,line.split())
            assert c in (0,1) and 0<x<1 and 0<y<1 and 0<bw<=1 and 0<bh<=1
            box=((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h)
            if min(box)<-2 or box[2]>w+2 or box[3]>h+2:
                print('OUT_OF_BOUNDS',start+j,r['name'],k,box,flush=True)
            d=ImageDraw.Draw(im)
            d.rectangle(box,outline='red' if c==0 else 'blue',width=max(3,w//240))
            d.text((box[0],box[1]),f'{k}:class{int(c)}',fill='red' if c==0 else 'blue',stroke_width=2,stroke_fill='white')
            annotations.append(int(c))
        im=ImageOps.contain(im,(504,450))
        x=j%3*512;y=j//3*500
        sheet.paste(im,(x,y))
        draw.text((x+3,y+453),f'{start+j}: {r["name"]} IDs:{annotations}',fill='black')
    sheet.save(RUN / f'preview-{start:03d}.jpg',quality=90)
