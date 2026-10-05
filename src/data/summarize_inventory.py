import pandas as pd

inventory = pd.read_csv("data/manifests/class_inventory.csv")
targets = pd.read_csv("data/manifests/data_targets.csv")

# Empty strict-match counts count as 0 for now
inventory["strict_match_images"] = pd.to_numeric(
    inventory["strict_match_images"],
    errors="coerce",
).fillna(0)

# Add up approved strict-match images for each target class
totals = (
    inventory.groupby("target_class")["strict_match_images"]
    .sum()
    .reset_index()
)

totals = totals.rename(
    columns={"strict_match_images": "current_strict_match_images"}
)

# Keep the class target and compare it to what we currently have
summary = targets[["target_class", "provisional_min_images"]].merge(
    totals,
    on="target_class",
    how="left",
)

summary["current_strict_match_images"] = (
    summary["current_strict_match_images"].fillna(0)
)

summary["deficit"] = (
    summary["provisional_min_images"]
    - summary["current_strict_match_images"]
)

summary["deficit"] = summary["deficit"].clip(lower=0)

print(summary.to_string(index=False))
