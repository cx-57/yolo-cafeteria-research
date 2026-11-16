import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


class RecoveryTests(unittest.TestCase):
    def test_preserves_primary_and_only_adds_strong_small_nonduplicates(self):
        from v5_recovery import recover_detections
        primary = [[.1,.1,.3,.4,.9,4], [.5,.5,.8,.8,.8,0]]
        expert = [[.1,.1,.3,.4,.95,4], [.6,.1,.8,.4,.8,4],
                  [.61,.1,.81,.4,.75,4], [0,0,1,1,.99,4],
                  [.1,.6,.2,.8,.6,4], [.1,.6,.2,.8,.99,5]]
        before = copy.deepcopy((primary, expert))
        result = recover_detections(primary, expert, class_id=4)
        self.assertEqual(result, primary + [[.6,.1,.8,.4,.4,4]])
        self.assertEqual((primary, expert), before)
        result[0][0] = 0
        self.assertEqual(primary, before[0])

    def test_weak_original_does_not_block_recovery(self):
        from v5_recovery import recover_detections
        primary = [[.1,.1,.3,.4,.02,4]]
        result = recover_detections(primary, [[.1,.1,.3,.4,.9,4]], class_id=4)
        self.assertEqual(result, primary + [[.1,.1,.3,.4,.45,4]])

    def test_rejects_invalid_rule_parameters(self):
        from v5_recovery import recover_detections
        for changes in [dict(class_id=True),dict(class_id=6),dict(max_area=0),
                        dict(min_confidence=float('nan')),dict(min_iou=0),
                        dict(confidence_multiplier=1.1)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                recover_detections([], [], **(dict(class_id=4) | changes))


if __name__ == '__main__': unittest.main()
