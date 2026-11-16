# Audit-only small-food follow-up

The user explicitly chose **“Keep auditing; no training or cleanup.”** The prepared trial remains unauthorized and unstarted. Generated data copies remain intact. No model weights, confidence thresholds, original annotations/splits or live configuration were changed.

## New-source result

The original unified model and current food fallback were checked on 18 revision-pinned, preselected AlphaTrash publisher-training images: 12 organic and 6 paper scenes. This is a new-source diagnostic, not a certified detection benchmark; publisher folder labels are coarse and the backdrop is repeated. All images were already visually reviewed before prediction.

At the unchanged 60% display threshold:

- Original: no visible food boxes on any of the 12 organic scenes.
- Food fallback: one of 12 organic scenes had a visible food box, on coconut at 62%.
- Neither produced a visible food box on the six paper scenes.
- Greens were incorrectly labeled wrapper 80% in scene 8; background hardware was labeled utensil 70% in scene 9 and the collection compartment hinged_container 80% in scene 12. These other-class outputs came unchanged from the primary model.
- Known recorded-pool exact-byte/pixel overlap: zero. Closest available perceptual fingerprint distance: 10 bits; this is only a screen and cannot exclude all historical/pretraining exposure.

[Fallback annotated contact sheet](../runs/sorter/small-food-generalization-20261115/new-source-review/audit-only-inference/fallback-contact-sheet.jpg) · [Original contact sheet](../runs/sorter/small-food-generalization-20261115/new-source-review/audit-only-inference/baseline-contact-sheet.jpg) · [Raw predictions](../runs/sorter/small-food-generalization-20261115/new-source-review/audit-only-inference/results.json)

Visual inspection confirms the food box in scene 9 mostly follows the coconut; the other boxes above are background or class confusions. The common container setting, greens-heavy sample and repeated appearances limit interpretation. This flags a generalization gap rather than establishing a global accuracy or proving statistical overfitting. It gives no basis for claiming reliable detection of tiny physical crumbs.

## Data separation

A protected audit registry now records the original 186 validation images, 248 test images, all 52 outside diagnostic images and current camera diagnostic frames as excluded from training. Test images were not scored. External/camera diagnostics are already exposed development material, so future reports must not present them as sealed independent tests. The AlphaTrash sample is reserved as diagnostic material after its training-source review was rejected; the initial candidate-selection history remains preserved.

The prepared 256-image subset has zero byte-hash or recorded-group intersection with that registry. It remains untrained and has no cleanup authorization. The registry is an audit artifact; it does not rewrite original dataset membership.

[Protected evaluation register](../runs/sorter/small-food-generalization-20261115/protected-evaluation-registry.json) · [Subset exclusion audit](../runs/sorter/small-food-generalization-20261115/prepared-trial-exclusion-audit.json) · [User scope decision](../runs/sorter/small-food-generalization-20261115/audit-only-decision.json)

## Training-data finding

The existing training pool has 45 small food boxes in 24 images from 10 recorded groups. Fifteen are clipped edges of a larger meal in one group. One food-positive group accounts for 189 of 1000 food-positive training images. These are coverage/concentration observations, not a license to relabel or delete source data. All originals were preserved.

A better next evaluation must distinguish food types, independent physical pieces/capture episodes, object scale, lighting and background. A successful single bread frame or many variants of the same piece cannot establish generalization. The fixed platform crop was already tested and worsened the bread result, so it remains off.

## Source attribution

[AlphaTrash publisher](https://github.com/Patipol-BKK/alphatrash-dataset), copyright (c) 2023 Patipol Tiyajamorn, MIT license; the source/license text and pinned revision are retained with the [manifest](../runs/sorter/small-food-generalization-20261115/new-source-review/source-manifest.json). Images were sampled from the publisher’s original training split; no publisher validation/test image was downloaded. Prediction overlays/contact sheets are derived copies.
