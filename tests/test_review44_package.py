"""Proof that original filename order, not GitHub ZIP order, owns draft boxes."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from package_review44_annotation_kit import assemble

SIZES=[('hinged_container_candidate',12,'commons'),('utensil',15,'commons'),
       ('wrapper',2,'commons'),('bottle',9,'trashbox'),('drink_carton',6,'trashbox')]

class AnnotationPackageTests(unittest.TestCase):
    def synthetic(self,folder):
        images=[]
        for hint,count,source in SIZES:
            for i in range(count):
                data=f'{source}-{hint}-{i}'.encode()
                digest=hashlib.sha256(data).hexdigest()
                name=f'images/{source}__{hint}_{i:03d}_{digest[:11]}.jpg'
                path=folder/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
                images.append(dict(path=name,source=source,source_hint=hint,
                                   sha256=digest,scoring_eligible=True))
        ordered=sorted(images,key=lambda x:(['hinged_container_candidate','utensil','wrapper',
                                            'bottle','drink_carton'].index(x['source_hint']),
                                            Path(x['path']).name))
        # Set individual box lengths so total=104, keep each original identity
        # identifiable from its number of food boxes.
        coords=[]
        for i in range(44):
            count=3 if i<16 else 2
            coords.append([i,[['food',[.05,.05,.5,.5]] for _ in range(count)]])
        # Simulate GitHub generated digest ordering which DOES NOT follow labels.
        images.sort(key=lambda x:(x['source'],x['sha256']))
        (folder/'manifest.json').write_text(json.dumps(dict(images=images)))
        return coords,ordered

    def test_draft_boxes_stay_with_original_images_across_orderings(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)
            coords,ordered=self.synthetic(folder)
            assemble(folder,coords)
            result=json.loads((folder/'manifest.json').read_text())
            attached={x['sha256']:len(x['draft_boxes']) for x in result['images']}
            for i,r in enumerate(ordered):
                self.assertEqual(3 if i<16 else 2,attached[r['sha256']])
            self.assertEqual(104,sum(attached.values()))
            self.assertIn('canvas', (folder/'index.html').read_text())
            starter=json.loads((folder/'cafeteria-review-export.json').read_text())
            self.assertEqual(0,starter['reviewed_image_count'])
            self.assertFalse(any(x['reviewed'] for x in starter['images']))

    def test_rejects_renamed_photo_with_wrong_sha_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);coords,_=self.synthetic(folder)
            manifest=json.loads((folder/'manifest.json').read_text())
            manifest['images'][0]['path']=manifest['images'][0]['path'].replace('_'+
                 manifest['images'][0]['sha256'][:11], '_aaaaaaaaaaa')
            (folder/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'filename and SHA'):
                assemble(folder,coords)

if __name__=='__main__':
    unittest.main()
