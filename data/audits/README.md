# Visual dataset audits

Every source image in the seven uploaded datasets was visually inspected against the NHS cafeteria target domain.

| Source | Reviewed | Kept | Scrapped |
| --- | ---: | ---: | ---: |
| School Lunch v2 | 1,876 | 467 | 1,409 |
| Tetra Pak v2 | 1,710 | 661 | 1,049 |
| MacWorX Containers v7 | 420 | 121 | 299 |
| Utensil v1 | 465 | 128 | 337 |
| Sauce v2 | 59 | 0 | 59 |
| napkin v1 | 180 | 0 | 180 |
| Plastic Detector v11 | 115 | 0 | 115 |

The School Lunch re-upload was byte-identical to the previously processed source. Its previously selected candidates plus the remaining source images were all visually checked; the strict 467-image subset remains unchanged.

Detailed image-level KEEP/SCRAP CSVs and retained manifests are imported with the accepted image/label pairs by `scripts/import_reviewed_batch.sh`.

Some Roboflow YOLOv11 exports contain segmentation polygons. Retained polygon annotations are converted to axis-aligned YOLO detection boxes for this object-detection project.
