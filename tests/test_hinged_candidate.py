"""Tests for separate 3-class candidate construction. Synthetic labels only."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import hinged_candidate as h


def fake(root, name, classes):
    folder=root/'images'/'train'
    folder.mkdir(parents=True,exist_ok=True)
    image=folder/(name+'.jpg')
    image.write_bytes(('synthetic-'+name).encode())
    label=root/'labels'/'train'/(name+'.txt')
    label.parent.mkdir(parents=True,exist_ok=True)
    label.write_text(''.join(f'{c} 0.5 0.5 0.5 0.5\n' for c in classes))
    return dict(image=str(image),label=str(label),
                sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                label_sha256=hashlib.sha256(label.read_bytes()).hexdigest())


class HingedTests(unittest.TestCase):
    def make_info(self,root):
        train=[
            *[fake(root,f'hinge{i}',[3,0]) for i in range(6)],
            *[fake(root,f'carton{i}',[4]) for i in range(4)],
            *[fake(root,f'food{i}',[0]) for i in range(4)],
            *[fake(root,f'plate{i}',[1]) for i in range(4)],
            fake(root,'both',[3,4,1]) ]
        # Make a distinct separate val
        validation=[fake(root,'val_hinge',[3]),fake(root,'val_food',[0]),
                    fake(root,'val_carton',[4]),fake(root,'val_plate',[1])]
        return {'records':{'train':train,'val':validation}}

    def test_select_all_hinges_and_plate_negatives(self):
        with tempfile.TemporaryDirectory() as tmp:
            info=self.make_info(Path(tmp)/'src')
            selected,hinges,negs=h.choose_train(info['records']['train'],
                                                  seed=7,support_cap=2,background_cap=3)
            self.assertEqual(7,hinges)
            self.assertEqual(3,negs)
            self.assertEqual(14,len(selected))
            self.assertTrue(all(r['image'] not in {v['image'] for v in info['records']['val']} for r in selected))

    def test_three_class_remap_and_validation_not_repeated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            info=self.make_info(root/'src')
            output=root/'snapshot'
            report=h.export_dataset(info,output,repeat=3,support_cap=2,background_cap=2)
            self.assertEqual(7,report['train_hinged_views'])
            self.assertEqual(4,report['val_images'])
            config=__import__('yaml').safe_load((output/'data.yaml').read_text())
            self.assertEqual(['food','hinged_container','drink_carton'],config['names'])
            val_files=list((output/'labels'/'val').glob('*.txt'))
            self.assertEqual(4,len(val_files))
            self.assertEqual(1,sum('1 ' in x.read_text() for x in val_files))
            train_files=list((output/'labels'/'train').glob('*.txt'))
            self.assertEqual(7*3 + 2+2+2,len(train_files))
            self.assertTrue(all(int(row.split()[0])<3
                for p in train_files for row in p.read_text().splitlines()))
            self.assertEqual('',next(p for p in val_files if p.read_text()=='').read_text())

    def test_corrected_frozen_validation_overrides_stale_disk_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            info=self.make_info(root/'src')
            corrected={r['sha256']:[] for r in info['records']['val']}
            corrected[info['records']['val'][-1]['sha256']]=['3 0.5 0.5 0.5 0.5']
            out=root/'snapshot'
            h.export_dataset(info,out,repeat=1,support_cap=1,background_cap=1,
                             val_rows_by_sha=corrected)
            val_files=list((out/'labels'/'val').glob('*.txt'))
            self.assertEqual(1,sum(x.read_text().startswith('1 ') for x in val_files))
            self.assertEqual(3,sum(not x.read_text().strip() for x in val_files))

    def test_cannot_reuse_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            info=self.make_info(root/'src')
            out=root/'snapshot';out.mkdir()
            with self.assertRaises(FileExistsError):
                h.export_dataset(info,out)

    def test_invalid_repeat_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            info=self.make_info(root/'src')
            with self.assertRaises(ValueError):
                h.export_dataset(info,root/'snapshot',repeat=20)

    def test_no_hinge_cannot_be_trained(self):
        with tempfile.TemporaryDirectory() as tmp:
            info=self.make_info(Path(tmp)/'src')
            info['records']['train']=[r for r in info['records']['train']
                                      if not any(c==3 for c,_ in h.parse_rows(r['label']))]
            with self.assertRaisesRegex(ValueError,'hinged'):
                h.choose_train(info['records']['train'])

    def test_source_hint_is_not_accuracy(self):
        from compare_hinged_stress import hint_matches
        self.assertTrue(hint_matches('hinged_container_candidate',{'hinged_container'}))
        self.assertFalse(hint_matches('hinged_container_candidate',{'plate','food'}))
        self.assertFalse(hint_matches('wrapper',{'drink_carton'}))

    def test_extra_annotated_checks_and_train_only(self):
        import hashlib, zipfile
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            src=root/'extra.zip'
            img=b'fake-image-bytes'
            label=b'1 0.5 0.5 0.6 0.6\n'
            image_hash=hashlib.sha256(img).hexdigest()
            label_hash=hashlib.sha256(label).hexdigest()
            entries=[]
            with zipfile.ZipFile(src,'w') as z:
                for i in range(29):
                    photo=img+str(i).encode()
                    digest=hashlib.sha256(photo).hexdigest()
                    z.writestr(f'images/train/ex{i}.jpg',photo)
                    z.writestr(f'labels/train/ex{i}.txt',label)
                    entries.append(dict(image=f'images/train/ex{i}.jpg',
                        label=f'labels/train/ex{i}.txt',image_sha256=digest,
                        label_sha256=label_hash,source_gallery_index=i,
                        boxes=[dict(class_id=1)]))
                z.writestr('manifest.json',json.dumps(dict(
                    status='AI_VISUAL_DRAFT_NOT_HUMAN_VERIFIED',
                    classes=['food','hinged_container','drink_carton'],images=entries)))
            dest=root/'out';(dest/'images/train').mkdir(parents=True)
            (dest/'labels/train').mkdir(parents=True)
            r=h.append_new_annotated_train(src,dest,set(),set(),repeat=2)
            self.assertEqual(r['images_added'],29)
            self.assertEqual(r['training_views_added'],58)
            self.assertEqual(len(list((dest/'labels/train').glob('*.txt'))),58)
            with self.assertRaisesRegex(ValueError,'overlaps'):
                h.append_new_annotated_train(src,root/'out2',{entries[0]['image_sha256']},set(),repeat=2)

if __name__=='__main__':unittest.main()
