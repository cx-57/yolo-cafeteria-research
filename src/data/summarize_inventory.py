"""Summarize strict-match image counts and deficits from the dataset manifests."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

MANIFEST_DIR = Path("data/manifests")


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def summarize() -> pd.DataFrame:
    inventory = pd.read_csv(MANIFEST_DIR / "class_inventory.csv")
    targets = pd.read_csv(MANIFEST_DIR / "data_targets.csv")

    inventory["strict_match_images"] = _numeric(inventory["strict_match_images"])
    totals = (
        inventory.groupby("target_class", as_index=False)["strict_match_images"]
        .sum()
        .rename(columns={"strict_match_images": "current_strict_match_images"})
    )

    out = targets.drop(columns=["current_strict_match_images", "deficit"], errors="ignore")
    out = out.merge(totals, left_on="target_class", right_on="target_class", how="left")
    out["current_strict_match_images"] = _numeric(out["current_strict_match_images"])
    out["provisional_min_images"] = _numeric(out["provisional_min_images"])
    out["deficit"] = (
        out["provisional_min_images"] - out["current_strict_match_images"]
    ).clip(lower=0)
    return out


if __name__ == "__main__":
    print(summarize().to_string(index=False))
