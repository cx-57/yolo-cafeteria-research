"""Independent coordinate checks against known boxes and the actual YOLO data loader."""
from pathlib import Path
import hashlib,json,shutil,tempfile,unittest
import torch
from PIL import Image
from sahi.prediction import ObjectPrediction
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from adapter import predictions_for_validation

class ValidationCoordinates(unittest.TestCase):
    def test_known_non_square_anisotropic_resize_and_empty(self):
        p=ObjectPrediction(bbox=[10,20,60,90],category_id=0,category_name='bottle',score=.8)
        got=predictions_for_validation([p],((2.,3.),(5.,7.)),torch.device('cpu'))
        torch.testing.assert_close(got['bboxes'],torch.tensor([[35.,47.,185.,187.]]),rtol=0,atol=0)
        torch.testing.assert_close(got['conf'],torch.tensor([.8]),rtol=0,atol=0)
        self.assertEqual(got['cls'].tolist(),[0.])
        self.assertEqual(predictions_for_validation([],((1.,1.),(0.,0.)),torch.device('cpu'))['bboxes'].shape,(0,4))

    def test_original_boxes_align_with_real_validation_loader(self):
        data=Path('/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/data/bottle-capture-validation')
        manifest=json.loads((data/'manifest.json').read_text())
        record=next(r for r in manifest['records'] if r['split']=='val' and r['rows'])
        image=data/'images/val'/record['name'];label=data/'labels/val'/(image.stem+'.txt')
        before=(hashlib.sha256(image.read_bytes()).hexdigest(),label.read_bytes())
        with Image.open(image) as im:width,height=im.size
        objects=[]
        for row in label.read_text().splitlines():
            c,x,y,w,h=map(float,row.split());self.assertEqual(c,0)
            objects.append(ObjectPrediction(bbox=[(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height],category_id=0,score=.8))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'images').mkdir();(root/'labels').mkdir()
            shutil.copy2(image,root/'images'/image.name);shutil.copy2(label,root/'labels'/label.name)
            ds=YOLODataset(img_path=str(root/'images'),imgsz=640,batch_size=1,augment=False,hyp=get_cfg(),rect=True,cache=False,data={'names':{0:'bottle'},'nc':1},stride=32,pad=.5)
            batch=ds.collate_fn([ds[0]])
            got=predictions_for_validation(objects,batch['ratio_pad'][0],torch.device('cpu'))['bboxes']
            x,y,w,h=batch['bboxes'].T;ih,iw=batch['img'].shape[2:]
            expected=torch.stack(((x-w/2)*iw,(y-h/2)*ih,(x+w/2)*iw,(y+h/2)*ih),dim=1)
            torch.testing.assert_close(got,expected,rtol=0,atol=.002)
        self.assertEqual((hashlib.sha256(image.read_bytes()).hexdigest(),label.read_bytes()),before)

if __name__=='__main__':unittest.main()
