# School Lunch v2 — manually reviewed NHS-compatible subset

Source: https://universe.roboflow.com/school-lunch-dataset/school-lunch-lty6a/dataset/2

License: CC BY 4.0

All 1,876 source images were visually reviewed. Only the strict NHS-transfer subset belongs here.

- Source images reviewed: 1,876
- Images retained: 467
- Images scrapped: 1,409
- Final project class IDs used in labels: `0 = food`, `7 = drink_carton`
- Re-upload verified byte-identical to the previously processed Roboflow source

The source train/validation split is recorded in `data/manifests/school_lunch.csv`; processed files are stored flat so the project can create its own final split later.
