# Optional live food fallback

An explicit `--detector food-fallback` mode now uses the existing frozen food specialist only when the original unified model has no food at the configured display threshold. Confident original food detections are preserved; the other six classes come entirely from the original model. Native expert IDs are mapped by verified class names. Both checkpoints remain unchanged.

This is a detection-only candidate. Manual and autonomous hardware modes are rejected before model loading. The default remains `unified`. There was no retraining, confidence tuning, label/split change or camera/servo operation.

## Run it

Use the currently configured camera:

```sh
python3 scripts/live_sorter.py --detector food-fallback --mode detection --device mps
```

Use `--device cpu` if MPS is unavailable. Camera identity and ROI still use the existing controls; no external-webcam index or loading area is guessed. To run the original model, omit `--detector food-fallback`.

Image-only smoke test (no camera or serial):

```sh
python3 scripts/live_sorter.py --detector food-fallback \
  --image runs/sorter/external-object-check-20261115/images/rice.jpg \
  --device cpu
```

## What the outside photos show

[Original versus fallback: bowl bread, rice, oranges](../runs/sorter/external-object-check-20261115/food-expert-comparison/fallback-before-after.jpg)

Across 22 externally sourced photos selected before each batch inference, actual adapter checks passed on CPU and MPS. All confident original food outputs and all other-six raw predictions were preserved. The expert was consulted on 13 of 22 photos. These scene-only visual checks are provisional, without certified object boxes.

| Scene | Original at 60% | Fallback at 60% | Interpretation |
|---|---|---|---|
| Bowl of bread pieces | No food | Food 70% | Recovers the food group, not individual pieces |
| Rice meal | Plate only | Food 79% plus unchanged plate | Recovers food on meal; rice grains are not separate boxes |
| Orange pieces | Food 69% | Same Food 69% | Avoids the unconditional specialist regression |
| Loose and rye crumbs in second batch | Food 79% / 88% | Original detections preserved | No new crumb gain established |
| Cookie crumbs on plate | Whole plate labeled food 89% | Same incorrect broad box | Unresolved localization error |
| Ground breadcrumbs in metal bowl | No food | No food | Still missed in second batch |
| Isolated orange cross-section | No food | No food | Still missed in second batch |
| Grouped cutlery and wrapper scenes | Missed | Same outputs | Food fallback does not repair these classes |

These photos are now exposed diagnostic material. They were not used for training, and are not a sealed future test. No exact-byte/pixel matches were found with the recorded local combined pool; historical/pretraining exposure remains unknown.

## Fixed development validation

Actual adapter predictions on all 186 existing development images match fresh food-expert routing verification to 1e-5. At confidence 0.60 and box IoU 0.50:

| Model | Food TP | Food FP | Food FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Original |293|6|50|97.99%|85.42%|
| Fallback |295|6|48|98.01%|86.01%|

All other-six predictions, operating-point counts and per-class AP50/AP50–95 are identical to the original. This passes the predeclared food/other-class development nonregression check. This cohort is historically exposed, provisionally AI-labeled, and contains zero food boxes below 32² pixels at 640. It cannot establish independent live crumb accuracy. No held-out-test inference was performed.

## Latency

Both models were warmed on all 22 static photo aspect ratios, followed by three paired repetitions with alternating order. These are inference-wrapper timings, not live camera/display FPS. Initial device/shape warmup takes longer.

| Device | Original mean | Fallback mean | Fallback 95th percentile |
|---|---:|---:|---:|
| cpu | 46.4ms | 72.0ms | 109.2ms |
| mps | 17.7ms | 27.2ms | 39.1ms |

The expert runs on frames where food is absent at the display threshold, including some empty scenes, so it adds latency. Keeping this opt-in allows a live comparison without replacing the default.

## Verification

- 222 tests run: 221 passed, one existing skip. Nine new fallback tests passed.
- Review identified a native-ID assumption. A regression test first failed, then passed after mapping food by verified native name.
- Full image-only CLI smoke test saved a food 79% result on rice.
- Actual adapter validated all 186 development images and 22 external photos; CPU/MPS external parity checks preserved confident original food and other-six outputs.
- Original checkpoint, frozen specialists, active sorter configuration and model-ownership profiles passed SHA256 preservation checks.

[Adapter validation](../runs/sorter/external-object-check-20261115/food-expert-comparison/fallback-live-adapter-validation.json) · [CPU/MPS adapter verification](../runs/sorter/external-object-check-20261115/food-expert-comparison/fallback-live-adapter-verification.json) · [Warm timing evidence](../runs/sorter/external-object-check-20261115/food-expert-comparison/fallback-warm-latency.json)

## Source photos and licenses

Initial 13-photo sources and credits are in [external diagnostic](external-object-diagnostic.md) and its [source manifest](../runs/sorter/external-object-check-20261115/reviewed-sources.json). Follow-up sources and all download errors are recorded [here](../runs/sorter/external-object-followup-20261115/reviewed-sources.json). The originally selected saucer/fork image was not downloadable; it was not tested. Overlay images and contact sheets retain their respective source attribution/share-alike requirements.

- [Miettes de pain.jpg](https://commons.wikimedia.org/wiki/File:Miettes_de_pain.jpg) — Jeangagnon; [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0).
- [Roggenvollkornbrot.Kruemel.jpg](https://commons.wikimedia.org/wiki/File:Roggenvollkornbrot.Kruemel.jpg) — Membeth; [CC0](http://creativecommons.org/publicdomain/zero/1.0/deed.en).
- [Breadcrumb.jpg](https://commons.wikimedia.org/wiki/File:Breadcrumb.jpg) — Yvwv; [CC BY-SA 3.0](http://creativecommons.org/licenses/by-sa/3.0/).
- [Bread crumbs.jpg](https://commons.wikimedia.org/wiki/File:Bread_crumbs.jpg) — Tiia Monto; [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0).
- [Orange slice.jpg](https://commons.wikimedia.org/wiki/File:Orange_slice.jpg) — Flik R (flikr); [CC BY 2.0](https://creativecommons.org/licenses/by/2.0).
- [Orange slice - right.jpg](https://commons.wikimedia.org/wiki/File:Orange_slice_-_right.jpg) — Adrià from Lleida, Catalunya; [CC BY 2.0](https://creativecommons.org/licenses/by/2.0).
- [Stainless steel Apostle spoon.jpg](https://commons.wikimedia.org/wiki/File:Stainless_steel_Apostle_spoon.jpg) — Smirkybec; [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0).
- [Egg spoon stainless steel.jpg](https://commons.wikimedia.org/wiki/File:Egg_spoon_stainless_steel.jpg) — Kjersti Lie; [CC BY 3.0](https://creativecommons.org/licenses/by/3.0).
- [CoCo Ichibanya Spoon and fork.jpg](https://commons.wikimedia.org/wiki/File:CoCo_Ichibanya_Spoon_and_fork.jpg) — Corpse Reviver; [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0).

## Additional camera and stress check

[Extended check](food-fallback-extended-check.md): 12 additional photos (34 total), one newly recovered bread scene, zero visible food predictions on five clean plate/paper scenes; rice on a spoon and bread on a blue plate remain missed. The bounded live camera smoke test completed 425 frames in about 30 seconds at 14.14 processed FPS without errors. Recognition correctness still depends on observed objects and user feedback.
