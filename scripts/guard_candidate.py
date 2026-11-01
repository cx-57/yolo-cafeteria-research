"""Check a candidate's per-class validation retention before any model promotion.

Accepts existing specialists.py evaluate JSON files from the same frozen
validation manifest. This gate never trains, activates, or evaluates on test.
"""
import argparse
import json
import math
from pathlib import Path

METRICS = ("AP50", "AP50_95")


def check_candidate(baseline, candidate, target, minimum_gain=0.01,
                    max_regression=0.002):
    if (type(minimum_gain) not in (float, int)
            or type(max_regression) not in (float, int)
            or not math.isfinite(minimum_gain) or not math.isfinite(max_regression)
            or minimum_gain <= 0 or max_regression < 0):
        raise ValueError("Invalid target gain or regression tolerance")
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        raise ValueError("Both inputs must be JSON objects")
    reasons = []
    if not baseline.get("manifest_sha256"):
        reasons.append("Baseline missing frozen manifest fingerprint")
    if baseline.get("manifest_sha256") != candidate.get("manifest_sha256"):
        reasons.append("Evaluation manifest differs from baseline")
    if baseline.get("images") != candidate.get("images"):
        reasons.append("Evaluation image count changed")
    bclasses = baseline.get("per_class")
    cclasses = candidate.get("per_class")
    if not isinstance(bclasses, dict) or not isinstance(cclasses, dict):
        raise ValueError("Both inputs require per_class object metrics")
    if set(bclasses) != set(cclasses):
        reasons.append("Compared class sets differ")
    if target not in bclasses or target not in cclasses:
        raise ValueError("Target class not present in both metrics")
    delta = {}
    for name in sorted(set(bclasses) & set(cclasses)):
        b, c = bclasses[name], cclasses[name]
        if b.get("objects") != c.get("objects"):
            reasons.append(name + ": annotated object support changed")
        delta[name] = {}
        for metric in METRICS:
            old, new = b.get(metric), c.get(metric)
            if (type(old) not in (int, float) or type(new) not in (int, float)
                    or not math.isfinite(old) or not math.isfinite(new)
                    or not 0 <= old <= 1 or not 0 <= new <= 1):
                raise ValueError("Invalid " + metric + " for " + name)
            difference = new - old
            delta[name][metric] = round(difference, 8)
            if difference < -max_regression - 1e-10:
                reasons.append(name + ": " + metric + " regressed beyond tolerance")
    for metric in METRICS:
        old, new = baseline.get(metric), candidate.get(metric)
        if type(old) not in (int, float) or type(new) not in (int, float):
            raise ValueError("Both evaluations require aggregate " + metric)
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in (old, new)):
            raise ValueError("Invalid aggregate " + metric)
        if new - old < -max_regression - 1e-10:
            reasons.append("Aggregate " + metric + " regressed")
    if delta[target]["AP50_95"] < minimum_gain - 1e-10:
        reasons.append(target + ": AP50-95 improvement below minimum target gain")
    return dict(
        passed=not reasons, target=target,
        required_target_AP50_95_gain=minimum_gain,
        permitted_per_class_or_aggregate_regression=max_regression,
        baseline_images=baseline.get("images"),
        baseline_manifest_sha256=baseline.get("manifest_sha256"),
        delta=delta, blockers=reasons,
        caveat=("Validation-only development gate, not a claim of independent "
                "generalization. Do not train on test images or repeatedly "
                "select with test results. Promotion requires separate checks."),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path)
    p.add_argument("candidate", type=Path)
    p.add_argument("--target", required=True)
    p.add_argument("--minimum-gain", type=float, default=0.01)
    p.add_argument("--max-regression", type=float, default=0.002)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("Refusing to overwrite existing result")
    result = check_candidate(json.loads(a.baseline.read_text()),
                             json.loads(a.candidate.read_text()), a.target,
                             a.minimum_gain, a.max_regression)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
