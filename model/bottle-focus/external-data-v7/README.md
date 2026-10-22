# External bottle training supplement — October 22, 2026

Goal: improve the fixed bottle development benchmark beyond 82.65% AP50.
No new-data improvement has been established yet.

Two internet sources were inspected:

- Saifullah Anik, *YOLO-Formatted Dataset of Bottles in Diverse Backgrounds*, DOI [10.17632/tksn2jrckf.1](https://data.mendeley.com/datasets/tksn2jrckf/1), CC BY 4.0. Its 548 image/label pairs have public checksums. A deterministic 24-photo sample was downloaded and checksum verified. EXIF-aware visual review found many unannotated beverage bottles, repeated shop backgrounds, and two images with out-of-bounds boxes. None was admitted. Source metadata and the rejection report remain available.
- Gupta et al., [LVIS](https://github.com/lvis-dataset/lvis-api), official v1 training annotation archive SHA-256 `334a4caa374030a7817cf050364525e910f7960f9b6968cef47cffbf3893f8ba`, category 1162 `water_bottle`. Sixty new COCO/Flickr image links were attempted after excluding known source IDs; 37 decoded at the expected native dimensions, 13 failed download, and 10 had invalid dimensions/content. The source URLs, license IDs, and annotations are retained per image. Dataset/code licenses do not replace individual photograph licensing.

AI visual review selected 19 LVIS photos and 36 plastic drink-bottle boxes. These are **experimental training annotations, not human verified**. Reusable insulated bottles were excluded. The scene showing the same birthday gathering as existing validation photographs was explicitly held, even though the automatic feature screen missed that capture relationship.

All 37 usable candidates were checked against all 246 fixed validation photographs using exact byte/decoded-pixel identity, rotation-aware perceptual hashes, and mutual SIFT/RANSAC matches. No automatic hold fired. A re-encoded validation-image control produced 125 mutual matches / 125 inliers and was detected. These checks are heuristic and do not prove complete capture independence. Known LVIS image IDs were also excluded before download.

The new dataset has 1,075 unique training photos: all 1,056 originals plus 19 additions. Existing context-positive photos remain sampled four times, other originals once, and each new photo four times, giving 1,504 exposures. The original validation image bytes, labels, membership, and 67 target boxes remain unchanged. Original data and checkpoints are retained. Images are cached outside the Desktop; this directory stores hashes, source metadata, decisions, and scripts.

Run with the existing local environment:

```sh
PYTHONPATH=/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/py312:$PWD/scripts .venv/bin/python model/bottle-focus/external-data-v7/build.py
PYTHONPATH=/Users/Charles1/Library/Caches/yolo-cafeteria-runtime/py312:$PWD/scripts BOTTLE_PHASE_EPOCHS=6 .venv/bin/python -u model/bottle-focus/external-data-v7/run.py
```

The detector starts from the immutable verified YOLO11s bottle checkpoint. Later feature layers adapt at AdamW learning rate 0.00001; pretrained BatchNorm running statistics remain fixed. No independent NHS test result is claimed. LVIS photos overlap the COCO pretraining domain and are training supplements only.
