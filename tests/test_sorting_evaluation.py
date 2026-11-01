import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_sorting as e


def obj(box, category):
    return dict(box_xyxy_normalized=box, category=category)


def pred(box, category, confidence):
    return dict(box_xyxy_normalized=box, category=category,
                detection_confidence=confidence)


class SortingEvaluationTests(unittest.TestCase):
    def test_correct_wrong_review_missed_and_extra_are_distinct(self):
        boxes = [
            [0, 0, .2, .2],
            [.25, 0, .45, .2],
            [.5, 0, .7, .2],
            [.75, 0, .95, .2],
        ]
        truth = {"images": {"a.jpg": [
            obj(boxes[0], "compost"), obj(boxes[1], "recycling"),
            obj(boxes[2], "landfill"), obj(boxes[3], "compost")]}}
        choices = {"decisions": {"a.jpg": [
            pred(boxes[0], "compost", .95),
            pred(boxes[1], "landfill", .9),
            pred(boxes[2], "review", .8),
            pred([0, .5, .2, .7], "recycling", .7)]}}
        result = e.evaluate(choices, truth)
        self.assertEqual(result["correct_automatic"], 1)
        self.assertEqual(result["wrong_bin_automatic"], 1)
        self.assertEqual(result["matched_review"], 1)
        self.assertEqual(result["missed_objects"], 1)
        self.assertEqual(result["unmatched_predictions"]["recycling"], 1)
        self.assertAlmostEqual(result["automatic_coverage_of_objects"], .5)
        self.assertAlmostEqual(result["accuracy_among_matched_automatic_decisions"], .5)
        self.assertAlmostEqual(result["correct_automatic_over_all_labeled_objects"], .25)
        self.assertAlmostEqual(result["precision_including_unmatched_auto_predictions"], 1/3)
        self.assertEqual(result["confusion_by_true_bin"]["recycling"]["landfill"], 1)

    def test_does_not_match_twice_and_counts_empty_image_false_positive(self):
        b = [.1, .1, .5, .5]
        truth = {"images": {"a.jpg": [obj(b, "compost")], "b.jpg": []}}
        choices = {"decisions": {"a.jpg": [
            pred(b, "compost", .9), pred(b, "compost", .8)],
            "b.jpg": [pred(b, "landfill", .7)]}}
        result = e.evaluate(choices, truth)
        self.assertEqual(result["correct_automatic"], 1)
        self.assertEqual(result["unmatched_predictions"]["compost"], 1)
        self.assertEqual(result["unmatched_predictions"]["landfill"], 1)
        self.assertAlmostEqual(result["precision_including_unmatched_auto_predictions"], 1/3)

    def test_images_cannot_be_dropped(self):
        truth = {"images": {"a.jpg": [], "b.jpg": []}}
        with self.assertRaisesRegex(ValueError, "match"):
            e.evaluate({"decisions": {"a.jpg": []}}, truth)

    def test_invalid_truth_or_predictions_rejected(self):
        b = [.1, .1, .5, .5]
        for category in ("review", "unknown"):
            with self.assertRaises(ValueError):
                e.evaluate({"decisions": {"x": []}},
                           {"images": {"x": [obj(b, category)]}})
        with self.assertRaises(ValueError):
            e.evaluate({"decisions": {"x": [pred(b, "compost", float("nan"))]}},
                       {"images": {"x": [obj(b, "compost")]}})
        with self.assertRaises(ValueError):
            e.evaluate({"decisions": {"x": []}},
                       {"images": {"x": [obj([0, 0, 1.1, 1], "compost")]}})


if __name__ == "__main__":
    unittest.main()
