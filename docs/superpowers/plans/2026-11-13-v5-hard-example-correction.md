# Correct confirmed V5 detection failures

User requested direct correction of missed cartons, wrappers and wooden cutlery. Prior generic fine-tunes did not resolve them. Treat diagnosed error images as training corrections, explicitly retiring their photo groups from unseen-test claims. Keep existing models, source datasets, old evaluation receipts and deployment untouched.

Work directory: `/Users/Charles1/Desktop/yolo-cafeteria-research-assets/artifacts/v5-hard-example-correction-20261113`.

## Data and evidence

- Fixed confidence0.25 / IoU0.50 baseline: cartons0/2, wrappers2/13, utensils12/20, including0/3 wooden utensils.
- Freeze original15 grouping: USDA20221028 public04–06, Japanese meal family public07–10, wooden cutlery public14, takeaway public15 move to correction-training side. Remaining public01,02,03,11,12,13 remain diagnostic checks, not certified independent field tests. No accuracy comparison on all15 after these examples enter training.
- Use existing124-box provisional annotation version2. Class-specific views never infer unseen objects. Verify originals/hash/box validity; retain source/license manifests. Corrected original training labels:74 utensil /29 wrapper boxes in separate copies.
- Carton correction positives04,06 and negative05; wrapper examples06,07,08,10,15 plus negative05; utensil examples07–10 and14. Full group exclusion even when only some frames are used.
- Mix64 original target-positive and64 original target-negative replay images, reviewed label corrections, earlier4 carton coverage photos, and8x exposures of direct correction examples. Exposures are not independent images.

## Bounded experiments

1. Carton: selected original projected V1 carton checkpoint;640px, three epochs, frozen backbone, train neck/head with fixed BN.
2. Wrapper: project selected unified wrapper branch;800px, three epochs, frozen backbone, train neck/head with fixed BN. Existing wrapper geometry policy stays fixed for evaluation.
3. Utensil: actual saved large V1 module architecture preserved, including custom pooling;800px, three epochs, backbone/neck frozen, train box and classifier head with fixed BN. Start after small jobs if resources require.
4. Exact initial transfer, checkpoint/source SHA, immutable dataset receipts and frozen tensors verified. CPU, no image copying, disk guards. No simultaneous large model jobs.
5. Predeclare retention: owned-class development AP50 and AP50–95 must each stay within2pp of selected baseline. Correction-learning checks at fixed confidence0.25/IoU0.50 are reported explicitly as training-example results. Remaining diagnostic cohorts reported separately; small support cannot establish readiness. Do not promote a regression or claim field performance from training replay.
6. Save predictions, per-image TP/FP/FN report, annotation-only and model overlays, checkpoint manifests, rejected outcomes and reproducible driver. Report whether concrete missed objects were actually learned.
