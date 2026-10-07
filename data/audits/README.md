# Visual dataset audits

Every uploaded source image was visually inspected against the NHS cafeteria target domain before inclusion.

Current audit result:

| Source | Reviewed | Kept | Scrapped |
| --- | ---: | ---: | ---: |
| School Lunch v2 | 1,876 | 467 | 1,409 |
| Tetra Pak v2 | 1,710 | 661 | 1,049 |
| MacWorX Containers v7 | 420 | 121 | 299 |
| Utensil v1 | 465 | 128 | 337 |
| Sauce v2 | 59 | 0 | 59 |
| napkin v1 | 180 | 0 | 180 |
| Plastic Detector v11 | 115 | 0 | 115 |

The School Lunch re-upload was byte-identical to the source that had already been manually audited, so its existing 467-image strict subset was left unchanged.

Detailed image-level KEEP/SCRAP CSVs and retained manifests are added by `scripts/import_reviewed_batch.sh` together with the accepted image/label pairs.
