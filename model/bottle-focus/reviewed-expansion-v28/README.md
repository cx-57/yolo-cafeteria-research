# Reviewed bottle expansion V28

V27 completed the hardware-adapted author recipe and scored 64.5471%
ordinary / 62.7979% augmented AP50. V28 instead starts from the immutable
84.0784% specialist (V23 alpha0.5, SHA 2cd1efe2…2289).

Use the V20 training exposures plus the 17 accepted round-four LVIS photos
with 20 source boxes, repeated four times each. This produces 1,115 unique
photos, 471 unique boxes and 1,964 training exposures. Prior external
additions comprise 57 photos / 109 boxes; cumulative external training is
74 photos / 129 boxes. Every added photo retains source URLs, license IDs,
source annotations, source hash, review and heuristic overlap-screen proof.
AI review is not human verification or proof of source independence.

Head-only AdamW 0.00001, freeze23, fixed BatchNorm, 640 pixels, batch4/nbs16,
seed43, mosaic0.5, scale0.5, no close-mosaic. The first phase is six epochs
of a 40-epoch schedule, preserving optimizer state. It is a screen, not a
convergence claim. Training validation and standard end-of-phase checks
use native NMS0.7; a separate freshly reloaded ordinary check uses NMS0.5.
All evaluations retain the frozen 246 validation photos and 67 boxes.
Only fresh saved-model improvement may replace the verified incumbent.

Two admission tests passed after RED, including changed source/screen/
annotation and duplicate-review rejection. The full project suite passed
235 tests with no skips. Independent review caught inherited addition
counts in the audit; these now explicitly distinguish new, prior and
cumulative additions.

Completed six epochs. Best native trained epoch 1 scored 82.3511% AP50;
fresh saved trained-model ordinary640/batch4/NMS0.5 scored 83.4063%.
The selected model remains initialization (83.3168% at NMS0.7; TTA 82.4785%).
The protected 84.0784% incumbent was not replaced. Phase 6 snapshots and
fresh trained-model verification are retained. This is a short screen,
not a claim of convergence or proof that the new photos cannot help.
