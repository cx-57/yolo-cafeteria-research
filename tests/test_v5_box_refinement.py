import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from v5_box_refinement import refine_geometry


class GeometryTests(unittest.TestCase):
    def test_agreement_changes_only_coordinates_not_confidence(self):
        base = [[.1, .1, .5, .5, .8, 5]]
        expert = [[.12, .1, .52, .5, .4, 5]]
        result = refine_geometry(base, expert)
        self.assertAlmostEqual(result[0][0], .11)
        self.assertAlmostEqual(result[0][2], .51)
        self.assertEqual(result[0][4:], [.8, 5])
        self.assertEqual(base[0][0], .1)
        self.assertEqual(expert[0][0], .12)

    def test_disagreement_low_confidence_and_wrong_class_leave_box_alone(self):
        base = [[.1, .1, .5, .5, .8, 5]]
        for expert in ([[.7, .7, .9, .9, .9, 5]],
                       [[.12, .1, .52, .5, .1, 5]],
                       [[.12, .1, .52, .5, .9, 0]]):
            self.assertEqual(refine_geometry(base, expert), base)

    def test_one_expert_cannot_move_two_objects(self):
        base = [[.1, .1, .5, .5, .9, 5], [.11, .1, .51, .5, .8, 5]]
        result = refine_geometry(base, [[.12, .1, .52, .5, .9, 5]])
        self.assertNotEqual(result[0][:4], base[0][:4])
        self.assertEqual(result[1], base[1])

    def test_other_classes_and_unmatched_expert_never_add_detections(self):
        base = [[.1, .1, .5, .5, .9, 0]]
        self.assertEqual(refine_geometry(base, [[.1, .1, .5, .5, .9, 5]]), base)
        self.assertEqual(refine_geometry([], [[.1, .1, .5, .5, .9, 5]]), [])

    def test_bad_parameters_or_boxes_are_rejected(self):
        for kwargs in ({'weight': 2}, {'min_iou': 0}, {'min_confidence': -1}):
            with self.assertRaises(ValueError):
                refine_geometry([], [], **kwargs)
        with self.assertRaises(ValueError):
            refine_geometry([[.5, .1, .1, .5, .9, 5]], [])


if __name__ == '__main__':
    unittest.main()
