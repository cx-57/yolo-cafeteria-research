import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import guard_candidate as g


class CandidateGuardTests(unittest.TestCase):
    def setUp(self):
        self.base = {
            "images": 30, "manifest_sha256": "frozen-labels",
            "AP50": .75, "AP50_95": .55,
            "per_class": {
                "food": {"AP50": .70, "AP50_95": .50, "objects": 50},
                "bottle": {"AP50": .80, "AP50_95": .60, "objects": 25},
            },
        }
        self.candidate = copy.deepcopy(self.base)
        self.candidate["AP50"] = .76
        self.candidate["AP50_95"] = .558
        self.candidate["per_class"]["bottle"]["AP50"] = .81
        self.candidate["per_class"]["bottle"]["AP50_95"] = .615

    def test_target_improves_without_other_regression(self):
        result = g.check_candidate(self.base, self.candidate, "bottle")
        self.assertTrue(result["passed"])
        self.assertEqual(result["blockers"], [])

    def test_other_class_regression_is_blocked(self):
        self.candidate["per_class"]["food"]["AP50_95"] = .49
        result = g.check_candidate(self.base, self.candidate, "bottle")
        self.assertFalse(result["passed"])
        self.assertTrue(any("food" in x for x in result["blockers"]))

    def test_candidate_must_retain_frozen_label_manifest(self):
        self.candidate["manifest_sha256"] = "changed-labels"
        self.assertFalse(g.check_candidate(self.base, self.candidate, "bottle")["passed"])
        self.candidate["manifest_sha256"] = "frozen-labels"
        self.candidate["per_class"]["bottle"]["objects"] = 24
        self.assertFalse(g.check_candidate(self.base, self.candidate, "bottle")["passed"])

    def test_target_gain_and_aggregate_both_required(self):
        self.candidate["per_class"]["bottle"]["AP50_95"] = .601
        self.assertFalse(g.check_candidate(self.base, self.candidate, "bottle")["passed"])
        self.candidate["per_class"]["bottle"]["AP50_95"] = .615
        self.candidate["AP50_95"] = .54
        self.assertFalse(g.check_candidate(self.base, self.candidate, "bottle")["passed"])

    def test_historical_retention_can_pass_without_extra_target_gain(self):
        old = copy.deepcopy(self.base)
        new = copy.deepcopy(self.base)
        self.assertTrue(g.check_candidate(old, new, "bottle",
                                          minimum_gain=0)["passed"])
        new["per_class"]["food"]["AP50_95"] -= .005
        self.assertFalse(g.check_candidate(old, new, "bottle",
                                           minimum_gain=0)["passed"])

    def test_invalid_metrics_rejected(self):
        self.candidate["per_class"]["bottle"]["AP50"] = float("nan")
        with self.assertRaises(ValueError):
            g.check_candidate(self.base, self.candidate, "bottle")
        with self.assertRaises(ValueError):
            g.check_candidate(self.base, self.base, "missing")


if __name__ == "__main__":
    unittest.main()
