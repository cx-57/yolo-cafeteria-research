# Gentle feature polishing

Stopped after three completed epochs: native AP50 declined 80.22% / 76.92% /
76.12%. Epoch four was interrupted and the optimizer through epoch three
is retained. The initial control remains selected; see phase3-outcome.json. Initialize from
verified-incumbent/bottle.pt and retain its hash and proof. Reuse the exact
v11 training list (1,092 unique photos / 1,623 exposures), all 41 external
photos / 80 boxes, five quality holds and 12 repeated reviewed negatives.
Validation remains 246 photos / 67 boxes, with no independent test claim.

Unfreeze layers 10 onward at AdamW learning rate 0.000003. All BatchNorm
running statistics remain fixed. Keep mosaic disabled and scale jitter 0.25.
Other settings match v11: 640 pixels, batch 4, nbs 16, MPS, seed 42,
six-epoch phases with optimizer continuation. Compare independently loaded
saved checkpoints to the starting model; retain the control separately.
