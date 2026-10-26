from pathlib import Path
from tempfile import TemporaryDirectory
import math
import unittest
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from dual import attach
from quality import install
from fusion import FlipFusionValidator
from test_dual import BASE,OTHER
class ValidatorSmoke(unittest.TestCase):
 def test_actual_autobackend_validator_fuses_every_synthetic_frame(self):
  torch.set_num_threads(2)
  with TemporaryDirectory() as folder:
   root=Path(folder);(root/'images/val').mkdir(parents=True);(root/'labels/val').mkdir(parents=True)
   for i in range(2):
    Image.fromarray(np.random.default_rng(i).integers(0,256,(96,128,3),dtype=np.uint8)).save(root/f'images/val/{i}.png');(root/f'labels/val/{i}.txt').write_text('0 .5 .5 .3 .6\n')
   config=root/'data.yaml';config.write_text(f'path: {root}\ntrain: images/val\nval: images/val\nnames: [bottle]\n')
   model=YOLO(BASE);secondary=YOLO(OTHER).model;model.model.fuse(verbose=False);secondary.fuse(verbose=False);install(model.model);install(secondary);trace=attach(model.model,secondary);finished=[]
   model.add_callback('on_val_end',lambda v:finished.append(v.fused_images))
   model.val(validator=FlipFusionValidator,data=str(config),imgsz=128,batch=2,device='cpu',workers=0,augment=True,iou=.5,conf=.001,plots=False,save_json=True,project=str(root),name='smoke')
   self.assertEqual(finished,[2]);self.assertEqual(trace['calls'],1);self.assertEqual(trace['shapes'][0][0],2)
if __name__=='__main__':unittest.main()
