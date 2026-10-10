# Trash Detection v1 — manually reviewed NHS-compatible subset

Source: https://universe.roboflow.com/blues-workspace-cpbsa/trash-detection-1fjjc-ntyvc/dataset/1

License: CC BY 4.0

Date reviewed: **October 7, 2026**

All 2,783 source images were screened. Potential bottle/wrapper candidates were visually audited in contact sheets, and the retained subset was reviewed again at larger scale.

Strict retention policy:

- `Clear plastic bottle` -> project class `10 = bottle`
- `Crisp packet` -> project class `9 = wrapper`
- at least one trusted target object must have normalized bounding-box area >= `0.005`
- `Other plastic bottle` rejected because it mixes beverage and household/chemical bottles
- `Other plastic wrapper` rejected because it mixes food wrappers, bags, film, and unrelated packaging
- all other source classes rejected for this replacement source
- retained segmentation polygons converted to axis-aligned YOLO detection boxes

Results:

- Source images screened: 2,783
- Images retained: 197
- Images scrapped: 2,586
- Retained object instances: 245
- Bottle instances: 208
- Wrapper instances: 37
- Native detection rows retained: 0
- Segmentation polygons converted: 245
