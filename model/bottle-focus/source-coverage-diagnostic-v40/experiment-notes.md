# Fresh source and scale diagnostic

Completed2026-10-25. No training, original benchmark inference or promotion.
Reserved32new source photos before any model measurements:16litter-positive photos/18boxes and16twice AI-reviewed absence photos. All positive boxes have area≥1024pixels at a uniform max-side640; the medium/large mix still differs from the original benchmark. Entire source groups and any aliases stay permanently excluded from training.

| Saved model | Fresh litter AP50 | Fresh litter AP50–95 |
|---|---:|---:|
| Balanced88877bbf… |47.560575%|28.203331%|
| V34,641d162e… |52.270221%|28.129373%|
| V36,4afd3108… |50.675168%|24.867968%|

The simple hypothesis that newer Open Images improvements accompanied worse litter AP50 was not confirmed. Localization remains weak for every model here, and the newer models have lower litter AP50–95 than the balanced initialization. Source coverage alone does not explain the original benchmark regressions. No causal attribution follows from these small, compositionally different cohorts.

The protected original benchmark remains246photos/67boxes. The balanced incumbent stays84.071068% AP50 /62.060566% AP50–95. These diagnostic scores cannot meet the user's90% benchmark move-on criterion. Both cohorts are public-source development monitors, not an independent NHS test; prior public-source model exposure is possible.

V42 prepares a larger reviewed-data full-training run from the balanced initialization with source-separated acceptance gates. The user's earlier pause before full training remains in force.
