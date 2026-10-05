"""End-to-end latency summaries."""

import numpy as np

def latency_summary(milliseconds: list[float]) -> dict[str, float]:
    if not milliseconds:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    arr = np.asarray(milliseconds, dtype=float)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
    }
