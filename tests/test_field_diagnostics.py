"""Unit tests for pure field diagnostics; do not load model weights."""
import sys
import tempfile
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import field_diagnostics as f


class FieldDiagnosticsTests(unittest.TestCase):
    def test_unlabeled_predictions_are_counts_not_accuracy(self):
        predictions = {'a.jpg': [[.1,.1,.4,.4,.90,0], [.2,.2,.5,.5,.35,6]], 'b.jpg': []}
        result = f.inspect_predictions(predictions)
        self.assertEqual(result['images'], 2)
        self.assertEqual(result['photos_without_displayed_detections'], 1)
        self.assertEqual(result['detections_at_025'], 2)
        self.assertEqual(result['detections_at_050'], 1)
        self.assertEqual(result['uncertain_025_to_050'], 1)
        self.assertEqual(result['per_class']['bottle']['detections'], 1)
        self.assertNotIn('precision', result)

    def test_cross_class_conflict_not_same_class(self):
        rows = {'a': [[0,0,1,1,.9,0], [0,0,1,1,.9,0], [0,0,1,1,.9,6]]}
        self.assertEqual(len(f.inspect_predictions(rows)['possible_cross_class_conflicts']), 2)

    def test_requires_human_review_attestation(self):
        with self.assertRaisesRegex(ValueError, 'labels-reviewed'):
            f.run('/not/needed','/not/needed',labels='labels')

    def test_missing_label_file_or_duplicate_stem(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            photo = root / 'one.jpg'; photo.write_bytes(b'')
            with self.assertRaisesRegex(ValueError, 'Missing COMPLETE'):
                f.read_complete_labels([photo], root)
            with self.assertRaisesRegex(ValueError, 'Duplicate image stems'):
                f.read_complete_labels([photo, root/'other'/'one.jpg'],root)

    def test_empty_label_is_true_negative_only_if_reviewed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            photo = root / 'one.jpg'; photo.write_bytes(b'')
            (root/'one.txt').write_text('')
            self.assertEqual(f.read_complete_labels([photo],root)[0]['rows'],[])


if __name__ == '__main__':
    unittest.main()
