"""Validate source target boxes independently of download/photo geometry status."""
from pathlib import Path
import hashlib,json,math
from PIL import Image,ImageDraw,ImageOps
RUN=Path(__file__).resolve().parent

def box_valid(box,width,height):
 if len(box)!=4 or not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in box):return False
 x,y,w,h=box
 return width>0 and height>0 and 0<=x<x+w<=width and 0<=y<y+h<=height

def main():
 path=RUN/'downloads.json';rows=json.loads(path.read_text());results=[];crops=[]
 for i,r in enumerate(rows):
  if r['status']!='unreviewed_staged_only':results.append(dict(index=i,id=r['id'],accepted_geometry=False,reason='Photo/source/license download hold'));continue
  width,height=r['original']['width'],r['original']['height'];valid=all(box_valid(a['bbox'],width,height) for a in r['annotations'])
  results.append(dict(index=i,id=r['id'],accepted_geometry=valid,reason='All target boxes finite, positive and within image' if valid else 'Invalid target source bbox: whole-photo hold'))
  if not valid:continue
  for j,a in enumerate(r['annotations']):
   im=Image.open(r['image']).convert('RGB');assert im.size==(width,height);d=ImageDraw.Draw(im);x,y,w,h=a['bbox'];d.rectangle((x,y,x+w,y+h),outline='red',width=1)
   mx=max(30,w*.2);my=max(30,h*.2);im=im.crop((max(0,int(x-mx)),max(0,int(y-my)),min(width,int(x+w+mx)),min(height,int(y+h+my))))
   crops.append((i,j,ImageOps.contain(im,(395,395))))
 for start in range(0,len(crops),8):
  sheet=Image.new('RGB',(1600,850),'white');d=ImageDraw.Draw(sheet)
  for k,(i,j,im) in enumerate(crops[start:start+8]):
   x=k%4*400;y=k//4*425;sheet.paste(im,(x,y));d.text((x+3,y+400),f'{i} / target {j}',fill='black')
  sheet.save(RUN/f'target-crops-{start:03}.jpg',quality=95)
 report=dict(source_downloads_sha=hashlib.sha256(path.read_bytes()).hexdigest(),audit_code_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),records=results,accepted=sum(r['accepted_geometry'] for r in results),no_source_boxes_edited=True)
 (RUN/'geometry-audit.json').write_text(json.dumps(report,indent=2)+'\n');print('SOURCE_BOX_GEOMETRY',report['accepted'],len(rows))
if __name__=='__main__':main()
