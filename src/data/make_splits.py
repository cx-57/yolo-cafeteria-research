import argparse

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

parser = argparse.ArgumentParser()
parser.add_argument("manifest")
parser.add_argument("--group-col", default="scene_id")
parser.add_argument("--seed", type=int, default=57)
args = parser.parse_args()

data = pd.read_csv(args.manifest)

# 70% train, 30% held out
split1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=args.seed,
)

train_index, held_index = next(
    split1.split(data, groups=data[args.group_col])
)

data["split"] = ""
data.loc[data.index[train_index], "split"] = "train"

held = data.loc[data.index[held_index]]

# Split the held-out 30% evenly into validation and test
split2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=args.seed,
)

val_index, test_index = next(
    split2.split(held, groups=held[args.group_col])
)

data.loc[held.index[val_index], "split"] = "val"
data.loc[held.index[test_index], "split"] = "test"

data.to_csv(args.manifest, index=False)

print(data["split"].value_counts())
