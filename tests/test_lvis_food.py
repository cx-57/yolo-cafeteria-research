"""Checks for recovering official food boxes without replacing plate labels."""
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import lvis_food


class LvisFoodTests(unittest.TestCase):
    def test_streaming_arrays_cross_chunk_boundaries(self):
        content = '{"categories": [{"name":"pizza"}], "annotations": [{"id": 1}, {"id": 2}], "images": []}'
        self.assertEqual(list(lvis_food.iter_array(io.StringIO(content), "annotations", 7)),
                         [{"id": 1}, {"id": 2}])
        self.assertEqual(list(lvis_food.iter_array(io.StringIO(content), "images", 7)), [])

    def test_food_box_preserves_official_geometry_and_current_class(self):
        row = lvis_food.food_box([20, 10, 60, 40], 200, 100)
        self.assertEqual(row, "0 0.250000000 0.300000000 0.300000000 0.400000000")
        with self.assertRaises(ValueError):
            lvis_food.food_box([190, 10, 60, 40], 200, 100)

    def test_missing_array_does_not_silently_drop_annotations(self):
        with self.assertRaisesRegex(ValueError, "Missing JSON array"):
            list(lvis_food.iter_array(io.StringIO('{"images": []}'), "annotations", 5))

    def test_ingredients_inside_a_composite_food_are_not_duplicate_food_targets(self):
        annotations = [
            {"id": 1, "category_id": 10, "bbox": [0, 0, 100, 100]},
            {"id": 2, "category_id": 20, "bbox": [20, 20, 10, 10]},
            {"id": 3, "category_id": 20, "bbox": [110, 20, 10, 10]},
        ]
        kept = lvis_food.food_portions(annotations, {10: "pizza", 20: "tomato"})
        self.assertEqual([a["id"] for a in kept], [1, 3])


if __name__ == "__main__":
    unittest.main()
