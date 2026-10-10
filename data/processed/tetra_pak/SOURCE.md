# tetra_pak — manually reviewed NHS-compatible subset

Source: https://universe.roboflow.com/school-gchcr/tetra-pak/dataset/2

License: CC BY 4.0

Every source image was visually reviewed against the NHS cafeteria object standard. Only accepted image/label pairs belong here.

- Source images reviewed: 1710
- Images retained: 661
- Images scrapped: 1049
- Target class: `7 = drink_carton`
- Retained object instances: 738
- Segmentation polygons converted to detection boxes: 427
- Native detection boxes retained: 311

Large/family-size cartons, synthetic/mockups, flattened unrelated packaging, and other non-NHS-like carton forms were rejected.

