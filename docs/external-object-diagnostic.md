# External object diagnostic through the live inference path

Thirteen externally sourced photos were selected before prediction and checked using the original unified model through `live_sorter.predict`, the same function used on camera frames. CPU, image size 640, confidence 0.60 and raw occupancy confidence 0.15 were unchanged. No training, threshold tuning, weight replacement, camera access, or servo operation occurred.

The check reproduces the main problem: food predictions can cover a plate or a food group instead of individual pieces. Rice, bowl bread pieces, grouped cutlery and both wrapper scenes produced no visible target-class detections at 60%. Clementine and brownies were recognized; one spoon was detected in full, while another had only its bowl localized.

[Annotated contact sheet](../runs/sorter/external-object-check-20261115/detections-contact-sheet.jpg) · [Raw contact sheet](../runs/sorter/external-object-check-20261115/raw-contact-sheet.jpg)

## Every selected result

| Scene | Visible predictions | Visual inspection | Source |
|---|---|---|---|
| [cookie-crumbs](../runs/sorter/external-object-check-20261115/annotated/cookie-crumbs.jpg) | food 89% | Incorrect localization: food 89% box covers almost the entire plate. Individual crumbs are not localized. Do not count as successful crumb detection. | [Chocolate-chip-cookie-crumbs-on-plate.png](https://commons.wikimedia.org/wiki/File:Chocolate-chip-cookie-crumbs-on-plate.png) |
| [bread](../runs/sorter/external-object-check-20261115/annotated/bread.jpg) | food 85% | Food 85% box includes the bread and much of the plate; two bread pieces are not individually separated. | [Bread on plate.jpg](https://commons.wikimedia.org/wiki/File:Bread_on_plate.jpg) |
| [brownie](../runs/sorter/external-object-check-20261115/annotated/brownie.jpg) | food 75% | Food 75% box covers the brownie stack. Scattered individual crumbs are not separately detected. | [Brownie with crumbs on plate.jpg](https://commons.wikimedia.org/wiki/File:Brownie_with_crumbs_on_plate.jpg) |
| [breadcrumb-bowl](../runs/sorter/external-object-check-20261115/annotated/breadcrumb-bowl.jpg) | None | No visible box. Strongest food prediction 58%, below the unchanged 60% threshold. | [Bread Crumbs in Bowl.jpg](https://commons.wikimedia.org/wiki/File:Bread_Crumbs_in_Bowl.jpg) |
| [oranges](../runs/sorter/external-object-check-20261115/annotated/oranges.jpg) | food 69% | Food 69% box groups several orange pieces; it does not localize each piece and leaves part of the whole orange outside. | [Orange-Fruit-Pieces.jpg](https://commons.wikimedia.org/wiki/File:Orange-Fruit-Pieces.jpg) |
| [clementine](../runs/sorter/external-object-check-20261115/annotated/clementine.jpg) | food 81% | Food 81% box localizes the whole clementine reasonably closely. | [Clementine orange.jpg](https://commons.wikimedia.org/wiki/File:Clementine_orange.jpg) |
| [rice](../runs/sorter/external-object-check-20261115/annotated/rice.jpg) | plate 66% | Plate 66% visible; food 56% remains below threshold. Rice is not separately localized. | [Cooked rice 01.jpg](https://commons.wikimedia.org/wiki/File:Cooked_rice_01.jpg) |
| [forks-spoon](../runs/sorter/external-object-check-20261115/annotated/forks-spoon.jpg) | None | No visible boxes; utensil candidates top out near 38%. | [ForkForkSpoon.jpg](https://commons.wikimedia.org/wiki/File:ForkForkSpoon.jpg) |
| [spoon](../runs/sorter/external-object-check-20261115/annotated/spoon.jpg) | utensil 62% | Utensil 62% box covers the spoon, with a small confidence margin above threshold. | [Stainless steel spoon.jpg](https://commons.wikimedia.org/wiki/File:Stainless_steel_spoon.jpg) |
| [soup-spoon](../runs/sorter/external-object-check-20261115/annotated/soup-spoon.jpg) | utensil 71% | Utensil 71% box covers only the spoon bowl and misses most of the handle. Class is right but localization is incomplete. | [Stainless Steel Soup Spoon.jpg](https://commons.wikimedia.org/wiki/File:Stainless_Steel_Soup_Spoon.jpg) |
| [wood-cutlery](../runs/sorter/external-object-check-20261115/annotated/wood-cutlery.jpg) | None | No visible boxes. Incorrect food 45% and utensil 43% candidates are both below threshold. | [Wooden Tableware 03.jpg](https://commons.wikimedia.org/wiki/File:Wooden_Tableware_03.jpg) |
| [wrapper](../runs/sorter/external-object-check-20261115/annotated/wrapper.jpg) | None | No predictions at or above the 15% raw occupancy threshold. | [Sweet wrapper.jpg](https://commons.wikimedia.org/wiki/File:Sweet_wrapper.jpg) |
| [wrappers](../runs/sorter/external-object-check-20261115/annotated/wrappers.jpg) | None | No wrapper detection. Incorrect plate 26% stays below display threshold. | [Papiers de carambars (2882927200).jpg](https://commons.wikimedia.org/wiki/File:Papiers_de_carambars_(2882927200).jpg) |

## Scope and overlap screening

Downloaded bytes and decoded pixels had no exact matches in the recorded 2806-image combined pool (train/validation/test membership was screened; test predictions were not used). Nearest recorded difference-hash distances ranged 11–19 bits among records containing perceptual fingerprints. This screens known local data; it cannot prove absence from all historical/pretraining data. The selected photos are now exposed diagnostics, not an independent future test set.

These are static photos passed through the live prediction function. This does not establish live camera accuracy, FPS, temporal stability, or physical sorting behavior. Source-description scene expectations and assistant visual observations are provisional and have no ground-truth boxes, so no accuracy percentage or object AP is reported.

One originally selected file, Crumbs saucer fork.jpg, could not be downloaded due to server throttling and was not tested. Its download failure is recorded; no replacement was selected after inspecting model results.

## Attribution and licenses

- [Chocolate-chip-cookie-crumbs-on-plate.png](https://commons.wikimedia.org/wiki/File:Chocolate-chip-cookie-crumbs-on-plate.png) — Knoell8504; [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0). Prediction overlays and contact sheets are derived from these photos; retain the respective source license and attribution when sharing them.
- [Bread on plate.jpg](https://commons.wikimedia.org/wiki/File:Bread_on_plate.jpg) — stu_spivack; [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0). Prediction overlays and contact sheets are derived from these photos; retain the respective source license and attribution when sharing them.
- [Brownie with crumbs on plate.jpg](https://commons.wikimedia.org/wiki/File:Brownie_with_crumbs_on_plate.jpg) — Whitney; [CC BY 2.0](https://creativecommons.org/licenses/by/2.0). Prediction overlays and contact sheets are derived from these photos; retain the respective source license and attribution when sharing them.
- [Bread Crumbs in Bowl.jpg](https://commons.wikimedia.org/wiki/File:Bread_Crumbs_in_Bowl.jpg) — Alabama Extension; [CC0](http://creativecommons.org/publicdomain/zero/1.0/deed.en). Prediction overlays and contact sheets are derived from these photos; retain the respective source license and attribution when sharing them.
- [Orange-Fruit-Pieces.jpg](https://commons.wikimedia.org/wiki/File:Orange-Fruit-Pieces.jpg) — Evan-Amos; [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0). Prediction overlays and contact sheets are derived from these photos; retain the respective source license and attribution when sharing them.
- [Clementine orange.jpg](https://commons.wikimedia.org/wiki/File:Clementine_orange.jpg) — Petr Kratochvil; [CC0](http://creativecommons.org/publicdomain/zero/1.0/deed.en). Prediction overlays and contact sheets are derived from these photos; retain the respective source license and attribution when sharing them.
- [Cooked rice 01.jpg](https://commons.wikimedia.org/wiki/File:Cooked_rice_01.jpg) — Zakaria Tunsung; [CC0](http://creativecommons.org/publicdomain/zero/1.0/deed.en). Prediction overlays and contact sheets are derived from these photos; retain the respective source license and attribution when sharing them.
- [ForkForkSpoon.jpg](https://commons.wikimedia.org/wiki/File:ForkForkSpoon.jpg) — Jonathunder; [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0). Prediction overlays and contact sheets are derived from these photos; retain the respective source license and attribution when sharing them.
- [Stainless steel spoon.jpg](https://commons.wikimedia.org/wiki/File:Stainless_steel_spoon.jpg) — Janakiraman janaki; [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0). Prediction overlays and contact sheets are derived from these photos; retain the respective source license and attribution when sharing them.
- [Stainless Steel Soup Spoon.jpg](https://commons.wikimedia.org/wiki/File:Stainless_Steel_Soup_Spoon.jpg) — Paolomarco; [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0). Prediction overlays and contact sheets are derived from these photos; retain the respective source license and attribution when sharing them.
- [Wooden Tableware 03.jpg](https://commons.wikimedia.org/wiki/File:Wooden_Tableware_03.jpg) — PagesAndPathsPH; [CC0](http://creativecommons.org/publicdomain/zero/1.0/deed.en). Prediction overlays and contact sheets are derived from these photos; retain the respective source license and attribution when sharing them.
- [Sweet wrapper.jpg](https://commons.wikimedia.org/wiki/File:Sweet_wrapper.jpg) — Si Griffiths; [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0). Prediction overlays and contact sheets are derived from these photos; retain the respective source license and attribution when sharing them.
- [Papiers de carambars (2882927200).jpg](https://commons.wikimedia.org/wiki/File:Papiers_de_carambars_(2882927200).jpg) — Guillaume Capron from Issy les Moulineaux, France; [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0). Prediction overlays and contact sheets are derived from these photos; retain the respective source license and attribution when sharing them.

## Machine-readable evidence

- [Reviewed results and raw predictions](../runs/sorter/external-object-check-20261115/reviewed-results.json)
- [Source URLs, credits, licenses, hashes and pre-inference review](../runs/sorter/external-object-check-20261115/reviewed-sources.json)
- [Initial selection and download log](../runs/sorter/external-object-check-20261115/sources.json)

The original model remains selected. The current evidence supports gathering varied, reviewed annotations for food fragments and metal utensils, including correct tight boxes; it does not support promoting the failed fine-tuning trial or lowering the threshold to force detections.
