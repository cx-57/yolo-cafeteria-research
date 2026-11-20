"""Reviewed 44-image source validation and packaging tests."""

# --- test_reviewed44.py ---
"""Synthetic unit tests: 44-image human-review validation before model scoring."""
import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_reviewed44 import dataset_from_review, CLASSES, main

class Reviewed44Tests(unittest.TestCase):
    def test_cli_allow_ai_reviewed_argument_is_forwarded(self):
        argv=['evaluate_reviewed44.py', '--bundle', 'source.zip',
              '--review', 'checked.json', '--v3-config', 'candidate.json',
              '--output', 'metrics.json', '--allow-ai-reviewed']
        with patch('sys.argv',argv), patch('evaluate_reviewed44.compare') as compare:
            main()
        self.assertTrue(compare.call_args.args[-1])

    def setup_source(self,root):
        bundle=root/'images.zip'
        review={'format':'CAFETERIA_PROVISIONAL_HUMAN_REVIEW_V1','not_independent':True,'images':[]}
        meta={'classes':CLASSES,'images':[]}
        with zipfile.ZipFile(bundle,'w') as z:
            for i in range(44):
                name=f'images/photo{i:02}.jpg'
                data=f'test-fixture-photo-{i:02}'.encode()
                digest=hashlib.sha256(data).hexdigest()
                entry={'id':f'p{i}','path':name,'sha256':digest,
                       'source_hint':CLASSES[i%7],
                       'scoring_eligible':i not in (33,34,35)}
                meta['images'].append(entry)
                review['images'].append(dict(
                  id=entry['id'],path=name,sha256=digest,reviewed=True,
                  boxes=[dict(status='HUMAN_REVIEWED',class_name=CLASSES[i%7],
                    box_xyxy_normalized=[.1,.2,.8,.9])]))
                z.writestr(name,data)
            z.writestr('manifest.json',json.dumps(meta))
        return bundle,review

    def test_ai_visual_labels_require_explicit_diagnostic_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle,review=self.setup_source(root)
            for item in review['images']:
                item['reviewed']=False
                item['assistant_reviewed']=True
                for box in item['boxes']:
                    box['status']='AI_VISUAL_REVIEWED'
            with self.assertRaisesRegex(ValueError,'not human-confirmed'):
                dataset_from_review(bundle,review,root/'refuse')
            records,summary=dataset_from_review(bundle,review,root/'allowed',
                                                    allow_ai_review=True)
            self.assertEqual(41,len(records))
            self.assertEqual('AI_ASSISTANT_VISUAL_APPROXIMATE_NOT_HUMAN_VERIFIED',
                              summary['label_quality'])
            self.assertEqual({'assistant_visual_approximate':41},summary['review_methods'])
            self.assertTrue(all(not x['review_verified_by_user'] for x in records))
            with self.assertRaisesRegex(ValueError,'not human-confirmed'):
                review['images'][0]['assistant_reviewed']=False
                dataset_from_review(bundle,review,root/'refuse-incomplete',
                                    allow_ai_review=True)

    def test_ai_diagnostic_still_rejects_unreviewed_box(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle,review=self.setup_source(root)
            for item in review['images']:
                item['reviewed']=False
                item['assistant_reviewed']=True
                for box in item['boxes']:
                    box['status']='AI_VISUAL_REVIEWED'
            review['images'][3]['boxes'][0]['status']='AI_DRAFT_UNVERIFIED'
            with self.assertRaisesRegex(ValueError,'requires AI_VISUAL_REVIEWED'):
                dataset_from_review(bundle,review,root/'out',allow_ai_review=True)

    def test_refuses_draft_box(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle,review=self.setup_source(root)
            review['images'][3]['boxes'][0]['status']='AI_DRAFT_UNVERIFIED'
            with self.assertRaisesRegex(ValueError,'requires HUMAN_REVIEWED'):
                dataset_from_review(bundle,review,root/'out')

    def test_refuses_unreviewed_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle,review=self.setup_source(root)
            review['images'][5]['reviewed']=False
            with self.assertRaisesRegex(ValueError,'not human-confirmed'):
                dataset_from_review(bundle,review,root/'out')

    def test_scoring_exclusion_and_optional_all_44(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle,review=self.setup_source(root)
            subset,summary=dataset_from_review(bundle,review,root/'out')
            self.assertEqual(41,len(subset))
            self.assertEqual(3,len(summary['excluded_photo_ids']))
            complete,all_info=dataset_from_review(bundle,review,root/'all',include_excluded=True)
            self.assertEqual(44,len(complete))
            self.assertEqual(0,len(all_info['excluded_photo_ids']))
            self.assertEqual(44,len(set(x['sha256'] for x in complete)))

    def test_same_photos_in_different_order_still_match_by_exact_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle,review=self.setup_source(root)
            review['images'].reverse()
            rows,report=dataset_from_review(bundle,review,root/'out')
            self.assertEqual(41,len(rows))
            self.assertEqual(41,report['scored_photos'])
            self.assertEqual('review44-01-',rows[0]['name'][:12])

    def test_changed_photo_hash_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle,review=self.setup_source(root)
            review['images'].reverse()
            review['images'][0]['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'DIFFERENT photo bytes'):
                dataset_from_review(bundle,review,root/'out')

    def test_duplicate_review_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle,review=self.setup_source(root)
            review['images'][1]['sha256']=review['images'][0]['sha256']
            with self.assertRaisesRegex(ValueError,'DIFFERENT photo bytes'):
                dataset_from_review(bundle,review,root/'out')

    def test_image_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle,review=self.setup_source(root)
            review['images'][12]['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'DIFFERENT photo bytes'):
                dataset_from_review(bundle,review,root/'out')


# --- test_review44_package.py ---
"""Proof that original filename order, not GitHub ZIP order, owns draft boxes."""

from package_review44_annotation_kit import assemble

SIZES=[('hinged_container_candidate',12,'commons'),('utensil',15,'commons'),
       ('wrapper',2,'commons'),('bottle',9,'trashbox'),('drink_carton',6,'trashbox')]

class AnnotationPackageTests(unittest.TestCase):
    def test_annotation_classes_reuse_canonical_model_classes(self):
        import specialists
        from package_review44_annotation_kit import NAMES
        self.assertIs(NAMES, specialists.NAMES)

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
