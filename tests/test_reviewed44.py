"""Synthetic unit tests: 44-image human-review validation before model scoring."""
import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_reviewed44 import dataset_from_review, CLASSES

class Reviewed44Tests(unittest.TestCase):
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
            with self.assertRaisesRegex(ValueError,'AI draft'):
                dataset_from_review(bundle,review,root/'out')

    def test_refuses_unreviewed_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle,review=self.setup_source(root)
            review['images'][5]['reviewed']=False
            with self.assertRaisesRegex(ValueError,'not manually'):
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

if __name__=='__main__':
    unittest.main()
