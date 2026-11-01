import sys
import tempfile
import unittest
import random
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import research_audit as a


class ResearchAuditTests(unittest.TestCase):
    def test_freeze_rejects_changed_external_annotation_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root/'labels.json'
            manifest.write_text('{"rows": []}')
            freeze = dict(protected_sha256={}, manifests={
                'test': dict(path=str(manifest), sha256=a.s.sha(manifest))})
            a.verify_freeze(freeze, root)
            manifest.write_text('{"rows": ["changed"]}')
            with self.assertRaisesRegex(ValueError, 'test'):
                a.verify_freeze(freeze, root)

    def test_bin_format_search_records_hashes_but_does_not_assert_human_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'bins.csv').write_text('image,disposal_bin\na.jpg,recycling\n')
            (root/'classes.yaml').write_text('names: [compost, recycling, landfill]\n')
            (root/'objects.xml').write_text('<annotation><object><name>Plastic Bottle</name></object></annotation>')
            result = a.search_label_formats(root)
            self.assertEqual(result['counts'], dict(csv=1, xml=1, yaml=1))
            self.assertEqual(len(result['file_hashes']), 3)
            self.assertEqual(result['bin_class_candidates'], ['classes.yaml'])
            self.assertEqual(result['csv_header_candidates'][0]['path'], 'bins.csv')
            self.assertNotIn('verified_human_labels', result)

    def test_school_review_figures_are_not_original_candidate_photographs(self):
        self.assertTrue(a.is_candidate_path('data/school-submission-20261027/images/IMG_3403.jpg'))
        self.assertTrue(a.is_candidate_path('data/school-submission-20261031/originals/IMG_4066.HEIC'))
        self.assertFalse(a.is_candidate_path('data/school-submission-20261027/review/sheet.jpg'))
        self.assertFalse(a.is_candidate_path('artifacts/run/plot.png'))

    def test_perceptual_index_agrees_with_brute_force_radius_search(self):
        rng = random.Random(42)
        codes = [rng.getrandbits(64) for _ in range(100)]
        index = a.PerceptualIndex()
        for i, code in enumerate(codes):
            index.add(code, str(i))
        for query in [codes[0], codes[1]^31, codes[2]^255, rng.getrandbits(64)]:
            expected = min((query^code).bit_count() for code in codes)
            found = index.find([query], 5)
            self.assertEqual(found is None, expected > 5)
            if found:
                self.assertEqual(found['distance'], expected)

    def test_reencoding_is_pixel_duplicate_and_rotation_is_near_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = Image.new('RGB', (31, 23))
            image.putdata([((x*17+y*3)%256, (x*7+y*19)%256, (x*y)%256)
                           for y in range(23) for x in range(31)])
            image.save(root/'a.png', compress_level=0)
            image.save(root/'b.png', compress_level=9)
            image.transpose(Image.Transpose.ROTATE_90).save(root/'rotated.png')
            first, second, rotated = [a.fingerprint(root/p) for p in ['a.png','b.png','rotated.png']]
            self.assertNotEqual(first['sha256'], second['sha256'])
            self.assertEqual(first['pixel_sha256'], second['pixel_sha256'])
            self.assertEqual(a.hash_distance(first['dhashes'], rotated['dhashes']), 0)

    def test_capture_day_overlap_blocks_even_distinct_images(self):
        candidate = dict(sha256='unseen', pixel_sha256='different', groups=['school-day-2026-10-29'], dhashes=[0])
        reference = dict(sha256='trained', pixel_sha256='other', groups=['school-day-2026-10-29'], dhashes=[2**64-1])
        reasons = a.overlap_reasons(candidate, reference)
        self.assertIn('capture_or_group_overlap', reasons)
        self.assertNotIn('exact_bytes', reasons)

    def test_no_match_does_not_certify_independence(self):
        self.assertEqual(a.independence_status([], False), 'unresolved_history')
        self.assertEqual(a.independence_status(['exact_bytes'], True), 'exposure_or_duplicate')
        self.assertEqual(a.independence_status([], True), 'eligible_pending_human_provenance_review')

    def test_decode_failure_is_not_evidence_of_training_exposure(self):
        self.assertEqual(a.independence_status(['decode_unavailable_no_visual_screen']),
                         'unresolved_visual_screen')

    def test_fn_diagnostics_separate_threshold_localization_and_no_detection(self):
        truth = [[.1,.1,.3,.3,0], [.4,.4,.6,.6,0], [.7,.7,.9,.9,0]]
        predictions = [[.1,.1,.3,.3,.49,0], [.5,.4,.7,.6,.8,0]]
        result = a.error_diagnostics(truth, predictions, 0)
        self.assertEqual(result['false_negatives']['below_automatic_threshold'], 1)
        self.assertEqual(result['false_negatives']['localization_candidate'], 1)
        self.assertEqual(result['false_negatives']['no_same_class_overlap'], 1)
        self.assertEqual(result['false_positives']['localization_candidate'], 1)

    def test_duplicate_detections_are_not_new_true_positives(self):
        result = a.error_diagnostics([[.1,.1,.3,.3,6]],
                                    [[.1,.1,.3,.3,.9,6],[.1,.1,.3,.3,.8,6]],6)
        self.assertEqual(result['tp'], 1)
        self.assertEqual(result['false_positives']['duplicate_or_competing_box'], 1)


if __name__ == '__main__':
    unittest.main()
