"""Create grouped train/validation/test splits from an image manifest."""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def grouped_split(manifest: pd.DataFrame, group_col: str, seed: int = 57) -> pd.DataFrame:
    df = manifest.copy()
    first = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, holdout_idx = next(first.split(df, groups=df[group_col]))
    df["split"] = ""
    df.loc[df.index[train_idx], "split"] = "train"
    holdout = df.loc[df.index[holdout_idx]]
    second = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_rel, test_rel = next(second.split(holdout, groups=holdout[group_col]))
    df.loc[holdout.index[val_rel], "split"] = "val"
    df.loc[holdout.index[test_rel], "split"] = "test"
    return df

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--group-col", default="scene_id")
    parser.add_argument("--seed", type=int, default=57)
    args = parser.parse_args()
    df = pd.read_csv(args.manifest)
    grouped_split(df, args.group_col, args.seed).to_csv(args.manifest, index=False)

if __name__ == "__main__":
    main()
