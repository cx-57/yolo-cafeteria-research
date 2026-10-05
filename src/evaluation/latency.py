import numpy as np

def latency_summary(milliseconds):
    if len(milliseconds) == 0:
        return {
            "mean_ms": 0,
            "p50_ms": 0,
            "p95_ms": 0,
        }

    return {
        "mean_ms": np.mean(milliseconds),
        "p50_ms": np.percentile(milliseconds, 50),
        "p95_ms": np.percentile(milliseconds, 95),
    }
