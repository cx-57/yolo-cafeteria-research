"""Real trainer/EMA/checkpoint smoke on synthetic images, never benchmark data."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import geometry
from trainer import Trainer
SOURCE=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-reviewed-neck-monitor-v36/weights/verified-88877bbf13c02a202811b6ca0a1ccdc2bde70b2916d366f4b67938df732d1518.pt')
class TrainerSmoke(unittest.TestCase):
 def test_real_training_and_saved_ema_keep_geometry_fixed(self):
  torch.set_num_threads(2);initial=YOLO(str(SOURCE)).model.float().cpu().eval();geometry.select(initial);reference=geometry.fixed_state(initial)
  classifier={n:p.detach().cpu().clone() for n,p in initial.named_parameters() if p.requires_grad};del initial
  with TemporaryDirectory() as directory:
   root=Path(directory);(root/'images/train').mkdir(parents=True);(root/'labels/train').mkdir(parents=True)
   rng=np.random.default_rng(43)
   for index in range(2):
    Image.fromarray(rng.integers(0,256,(96,96,3),dtype=np.uint8)).save(root/f'images/train/{index}.png');(root/f'labels/train/{index}.txt').write_text('0 .5 .5 .3 .6\n')
   config=root/'data.yaml';config.write_text(f'path: {root}\ntrain: images/train\nval: images/train\nnames: [bottle]\n')
   model=YOLO(str(SOURCE));checked=[]
   def saved(trainer):
    geometry.verify_fixed(trainer.model,reference);geometry.verify_fixed(trainer.ema.ema,reference)
    checkpoint=YOLO(str(trainer.last)).model.float().cpu().eval();geometry.select(checkpoint);geometry.verify_fixed(checkpoint,reference)
    self.assertTrue(any(not torch.equal(p.detach().cpu(),classifier[n]) for n,p in trainer.model.named_parameters() if n in classifier));checked.append(True)
   model.add_callback('on_model_save',saved)
   model.train(trainer=Trainer,data=str(config),epochs=1,imgsz=96,batch=2,nbs=2,device='cpu',workers=0,amp=False,optimizer='AdamW',lr0=.001,warmup_epochs=0.,warmup_bias_lr=0.,freeze=23,mosaic=0.,scale=0.,translate=0.,plots=False,save=True,project=str(root),name='smoke',verbose=False)
   self.assertEqual(checked,[True])
if __name__=='__main__':unittest.main()
