"""Keep all monitor source groups, byte aliases and pixel aliases out of training."""
from pathlib import Path
from collections import Counter
import hashlib
from PIL import Image,ImageOps

def hashes(path):
 path=Path(path);digest=hashlib.sha256(path.read_bytes()).hexdigest()
 with Image.open(path) as opened:
  im=ImageOps.exif_transpose(opened).convert('RGB');pixel=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
 return digest,pixel

def split(exposures,groups,monitor):
 exposures=Counter(exposures);assert set(groups)==set(exposures)
 assert all(type(v) is int and v>0 for v in exposures.values()) and monitor
 assert all(isinstance(r['source_group'],str) and r['source_group'] for r in monitor)
 source_hashes=[hashes(r['source_image']) for r in monitor]
 assert len({digest for digest,pixel in source_hashes})==len(source_hashes)
 assert len({pixel for digest,pixel in source_hashes})==len(source_hashes)
 for record,(digest,pixel) in zip(monitor,source_hashes,strict=True):assert (digest,pixel)==(record['image_sha'],record['pixels'])
 held_groups={r['source_group'] for r in monitor};digests={a for a,b in source_hashes};pixels={b for a,b in source_hashes}
 checked={p:hashes(p) for p in exposures}
 for p,(digest,pixel) in checked.items():
  if digest in digests or pixel in pixels:held_groups.add(groups[p])
 removed=[p for p in exposures if groups[p] in held_groups]
 result=Counter({p:n for p,n in exposures.items() if p not in removed})
 assert result and all(checked[p][0] not in digests and checked[p][1] not in pixels for p in result)
 return result,removed
