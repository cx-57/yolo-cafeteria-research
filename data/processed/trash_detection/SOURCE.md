# Trash Detection v1 — manually reviewed NHS-compatible subset

Source: https://universe.roboflow.com/blues-workspace-cpbsa/trash-detection-1fjjc-ntyvc/dataset/1

License: CC BY 4.0

Date reviewed: **October 7, 2026**

All 2,783 source images were screened. Images with potentially useful bottle/wrapper classes were visually reviewed in contact sheets, and the retained subset was reviewed again at larger scale.

Strict retention policy:

- `Clear plastic bottle` -> historical stored class `10 = bottle` (training class `9`)
- `Crisp packet` -> historical stored class `9 = wrapper` (training class `8`)
- require at least one trusted target object with normalized bounding-box area >= 0.005
- reject `Other plastic bottle` as a source class because it mixes beverage bottles with household/chemical bottles
- reject `Other plastic wrapper` as a source class because it mixes food wrappers with bags, film, and unrelated plastic packaging
- reject every other source class for this replacement dataset
- convert retained segmentation polygons to axis-aligned YOLO detection boxes

Results:

- Source images screened: 2,783
- Images retained: 197
- Images scrapped: 2,586
- Retained object instances: 245
- Bottle instances: 208
- Wrapper instances: 37
- Native detection rows retained: 0
- Segmentation polygons converted: 245

The final processed subset contains only the accepted images and their matching project-format labels.
