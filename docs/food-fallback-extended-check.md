# Extended fallback check and live camera smoke test

The fallback was tested on 12 additional externally sourced photos selected before prediction, bringing the external diagnostic total to 34. Each new scene was visually inspected before running inference. This is a diagnostic collection, not an independent field benchmark.

The actual FoodFallbackDetector implementation matched the prescribed routing on all 12 new photos and preserved other-six outputs. It added a food 75% detection on bread partly covered by a soldering iron on a black surface. All confident original food detections were retained. Rice on a spoon and bread on a blue plate were still missed.

No visible food predictions appeared on five clean no-food scenes: two clean plates and three paper scenes. One plate with residual crumbs was excluded from this negative count because the food semantics are ambiguous. No claim of broad false-positive rate is made from five images. Plate/utensil detection on these scenes also remains weak; the fallback preserves those original outputs.

[New fallback contact sheet](../runs/sorter/external-object-stress-20261115/fallback-contact-sheet.jpg) · [Actual adapter evidence](../runs/sorter/external-object-stress-20261115/fallback-results.json)

## Camera test

A bounded 30-second detection-only run completed 425 processed frames with no error. Actual processed throughput was 14.14 FPS, including the intervals between processing results, versus mean inference time 35.9 ms. These are different measurements: camera acquisition, GUI rendering and waiting add time beyond inference. No serial port or servo operation was used; no camera images were automatically saved. Predicted classes were logged without image pixels, but recognition correctness is unverified without the user identifying the objects presented.

After the user explicitly requested the application be run, the regular live app was launched with `--detector food-fallback --mode detection --device mps` and left open for the user to test. Press **q** to exit. The earlier bounded timing does not certify accuracy in this later session.

[Bounded camera timing and predicted labels](../runs/sorter/live-food-fallback-20261115/timing-and-predictions.json) · [Mode details and earlier validation](food-fallback.md)

No weights, thresholds, original dataset, split membership or defaults changed during this additional check. No retraining occurred.

## Sources and attribution

All new photo source URLs, original/download hashes, credits and license URLs are recorded in the [source manifest](../runs/sorter/external-object-stress-20261115/reviewed-sources.json). Prediction overlays and contact sheets retain each photo’s attribution/share-alike requirements.

- [Empty plate with fork.jpg](https://commons.wikimedia.org/wiki/File:Empty_plate_with_fork.jpg) — SWT666; [CC BY 4.0](https://creativecommons.org/licenses/by/4.0).
- [Plate Poland 02.jpg](https://commons.wikimedia.org/wiki/File:Plate_Poland_02.jpg) — Abraham; [CC BY 4.0](https://creativecommons.org/licenses/by/4.0).
- [Midwinter (unidentified stripe pattern) tableware - 2025-04-14 - Andy Mabbett - 06.jpg](https://commons.wikimedia.org/wiki/File:Midwinter_(unidentified_stripe_pattern)_tableware_-_2025-04-14_-_Andy_Mabbett_-_06.jpg) — Andy Mabbett; [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0).
- [Crumpled Paper.png](https://commons.wikimedia.org/wiki/File:Crumpled_Paper.png) — Epicpanda65; [CC0](http://creativecommons.org/publicdomain/zero/1.0/deed.en).
- [Paperball.png](https://commons.wikimedia.org/wiki/File:Paperball.png) — Army1987; [CC BY-SA 3.0](http://creativecommons.org/licenses/by-sa/3.0/).
- [Wrinkled paper.jpg](https://commons.wikimedia.org/wiki/File:Wrinkled_paper.jpg) — Turinboy; [CC BY 2.0](https://creativecommons.org/licenses/by/2.0).
- [Boiled White rice.jpg](https://commons.wikimedia.org/wiki/File:Boiled_White_rice.jpg) — kamath_ln; [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0).
- [Cooked Rice.jpg](https://commons.wikimedia.org/wiki/File:Cooked_Rice.jpg) — Ananth Pattabiraman; [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0).
- [Boiled white rice with sauce.jpg](https://commons.wikimedia.org/wiki/File:Boiled_white_rice_with_sauce.jpg) — Meritkosy; [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0).
- [Bread waste, Philadelphiabrücke, Meidling.jpg](https://commons.wikimedia.org/wiki/File:Bread_waste,_Philadelphiabr%C3%BCcke,_Meidling.jpg) — Herzi Pinki; [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0).
- [Bread on blue plate from an angle.jpg](https://commons.wikimedia.org/wiki/File:Bread_on_blue_plate_from_an_angle.jpg) — Kansattica; [CC BY 4.0](https://creativecommons.org/licenses/by/4.0).
- [Bread on black table with resting soldering iron.jpg](https://commons.wikimedia.org/wiki/File:Bread_on_black_table_with_resting_soldering_iron.jpg) — Kansattica; [CC BY 4.0](https://creativecommons.org/licenses/by/4.0).
