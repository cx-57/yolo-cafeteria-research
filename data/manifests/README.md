# Dataset manifests

These files separate **published source counts** from **strict NHS-match counts**.

## Files

- `dataset_sources.csv` — one row per candidate public dataset.
- `class_inventory.csv` — source-class-to-target-class inventory.
- `data_targets.csv` — provisional minimum image targets and eventual deficits.
- `image_manifest.csv` — image-level provenance after data are downloaded.

## Count definitions

### original_images
Independent source photographs reported by the source. Augmented derivatives do not count.

### published_candidate_images
Images known to contain a source class that could map to one of our target classes. This is **not** the same as usable data.

### strict_match_images
Images retained after visual audit under the NHS domain-matching rules. This is the number used for deficit calculations.

### strict_match_instances
Retained object instances after visual filtering and annotation QA.

## Status values

- `pending_download_audit` — source/count verified; dataset still needs local inspection.
- `pending_visual_filter` — source class/count known; images still need NHS similarity review.
- `needs_local_class_count` — public source confirms the class but does not expose a trustworthy per-class count.
- `no_verified_source_yet` — targeted search/collection still required.
- `verified_strict` — visually audited and approved for Dataset V1.

Never fill `strict_match_images` from a dataset headline count.
