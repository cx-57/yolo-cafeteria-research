# Scope-screened feature adaptation

Stopped after two completed epochs: native AP50 declined 76.67% / 71.36%.
Epoch three was interrupted and optimizer state through epoch two retained;
the initial control remains selected. See phase2-outcome.json. The full 124-photo original context
positive cohort was reviewed with targeted object crops. Ten photos with
14 boxes have uncertain material/type (reusable cycling bottles, possible
glass drink containers, milk-style containers and handled milk jugs). These
are conservative image-level training holds, not ten confirmed label errors.
Original data and all 246 validation photos / 67 bottle boxes stay unchanged.

Retain the existing five negative-image quality holds and 57 reviewed
external photos / 109 boxes. There are 1,098 unique training photos, 1,611
exposures and 451 unique bottle boxes. Reviews are AI-assisted, not human
verification. No independent NHS test performance is established.

Start from the hash-bound pre-soup v6 checkpoint, retaining the separate
83.68% global incumbent. The v6 checkpoint's ordinary baseline is freshly
measured before training. Use freeze 10, fixed BatchNorm running statistics,
AdamW 0.00001, mosaic 0.5, scale 0.5, 640 pixels, batch 4, nbs 16, seed 42,
and six-epoch phases. This is an exploratory combined intervention.
