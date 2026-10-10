# macworx — manually reviewed NHS-compatible subset

Source: https://universe.roboflow.com/macworx/containers-yw7aw/dataset/7

License: CC BY 4.0

Every source image was visually reviewed against the NHS cafeteria object standard. Only accepted image/label pairs are stored here.

- Source images reviewed: 420
- Images retained: 121
- Images scrapped: 299
- Target class: `5 = hinged_container`
- Polygon annotations converted to detection boxes where necessary: 37
- Native detection boxes retained: 84

Rejected images are intentionally excluded from `data/processed/`.
