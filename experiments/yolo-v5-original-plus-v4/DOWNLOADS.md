# YOLO V5 — GitHub download status

Updated 2026-11-04. Links below only point to **files verified on GitHub**; unavailable binaries do **not** have fabricated URLs.

## Available now

- [Colab notebook — Worker A/B/C reviewed-patch merge and 7-class training](YOLO_V5_Worker_Patch_Merge_7Class_Colab.ipynb)
- [Annotation audit and three-way assignment status](ANNOTATION_AUDIT_STATUS.md)
- [Previous V5 training Colab notebook](YOLO_V5_Original_Plus_V4_7Class_Train.ipynb)

## Generated files that are NOT uploaded to GitHub yet

| File | Size (approx) | Recommended location |
|---|---:|---|
| `YOLO_V5_partial_annotation_audit320.zip` | 204 MiB | GitHub Release asset |
| `V5_Worker_A_402_Pending_Images.zip` | 13.9 MB | GitHub Release asset |
| `V5_Worker_B_425_Pending_Images.zip` | 20.1 MB | GitHub Release asset |
| `V5_Worker_C_425_Pending_Images.zip` | 30.8 MB | GitHub Release asset |
| `V5_Command_Center_Control_Files.zip` | 199 KB | GitHub Release asset |
| `V5_Worker_Reviewed_Patch_Templates.zip` | 82 KB | GitHub Release asset |
| `MASTER_ASSIGNMENT_INDEX.csv` | 399 KB | GitHub Release asset or repo file |

The source conversation contains sandbox downloads for these artifacts. GitHub Release assets are appropriate for the **204 MiB dataset** because GitHub blocks ordinary repository blobs above 100 MiB. These binary assets cannot be uploaded by the currently available GitHub connector. Do not link a guessed release-asset URL.

### To publish the binary files

1. Open [GitHub Releases](https://github.com/cx-57/yolo-cafeteria-research/releases) and create a new release.
2. Give it a version such as `v5-annotation-handoff-2026-11-04`.
3. Drop the seven files above into the release asset upload area and publish.
4. Once published, GitHub displays real downloadable asset links. Colab can then use those, or user-managed copies on Google Drive.

**Caution:** Worker input archives contain provisional unreviewed annotations, not completed worker-corrected labels. The 1,252 worker reviews must be integrated before final seven-class validation. The 97-val / 71-test split must remain frozen.
