# Bottle reviewed expansion implementation plan

> Execute inline with superpowers:executing-plans. Existing authorization is to continue bottle experiments toward 85% without repeated approval gates.

**Goal:** Test reviewed round-four data from the protected specialist, retaining the frozen 246-photo/67-box development benchmark.

**Architecture:** Reuse V20 training exposures and fixed validation; admit only the 17 accepted, hash-bound LVIS photos (20 boxes), repeated four times. Use a separate V28 cache and experiment directory. Initialize from immutable V23 alpha0.5 weights (84.0784% CPU ordinary), rather than the unsuccessful V27 fresh COCO run.

**Tech stack:** Existing Python 3.12, Ultralytics 8.4.175, Torch/MPS runtime; no dependencies added.

**Spec and approaches:** V27 completed 20 epochs, best saved ordinary AP50 64.5471%, TTA 62.7979%. Extending that fresh-start recipe is possible but currently much worse than the incumbent. Prefer incremental reviewed data from the incumbent. First test head-only AdamW 0.00001, fixed BatchNorm, batch 4/nbs16, 640 pixels, seed43, mosaic0.5, scale0.5, six-epoch phase of a 40-epoch schedule; retain optimizer. This is a screening phase, not convergence. Native validation uses the established default NMS0.7; fresh ordinary checks use NMS0.5 and are recorded separately. If it declines, retain initialization and inspect the trained checkpoint before deciding on feature adaptation.

## Constraints and review focus

- No changes to original images, labels, split membership or validation scope.
- Fail closed on changed reviews, annotations, duplicate-screen results, source image hashes, or unreviewed additions.
- Review is AI assisted, not human verification; overlap screening is heuristic.
- 1,115 unique training photos / 471 unique boxes / 1,964 exposures expected.
- Native training scores cannot establish success; reload a saved checkpoint at ordinary640, batch4, NMS0.5, record exact hash and 246/67 support.

## Tasks

- [x] Add tests of source admission and real box projection, including mismatched screen/image/annotation rejection. Observe RED, implement minimal admission helper, observe GREEN.
- [x] Build and verify the separate V28 training copy, proving validation and incumbent unchanged. Run the full project suite.
- [ ] Prepare pinned incumbent initialization and launch the six-epoch training phase with live trainability/BN checks.
- [ ] Freshly validate selected/trained saved weights; promote only verified improvement. Review and commit code, provenance and actual results.

## Ledger

- The user authorized continued implementation; implementation choices above are the agent's, not claims of an exact paper replication.

- Independent review found inherited addition counts; corrected to distinguish new17/20, prior57/109 and cumulative74/129. No exposure changes.
- The baseline candidate uses copyfile so a read-only immutable source does not make the mutable selection checkpoint read-only.
