"""Source annotation eligibility fixed before diagnostic model measurements."""
import math
MIN_AREA_AT640=1024.

def areas_at640(record):
 w,h=record['width'],record['height'];assert w>0 and h>0 and record['rows']
 ratio=640/max(w,h);areas=[]
 for row in record['rows']:
  values=list(map(float,row.split()));assert len(values)==5
  c,x,y,bw,bh=values
  if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite source box')
  assert c==0 and bw>0 and bh>0 and x-bw/2>=-1e-8 and y-bh/2>=-1e-8 and x+bw/2<=1+1e-8 and y+bh/2<=1+1e-8
  areas.append(bw*w*bh*h*ratio*ratio)
 return areas

def eligible(record):return all(a>=MIN_AREA_AT640 for a in areas_at640(record))
