# LVIS paper plate verification — October 10, 2026

Verified the uploaded official LVIS v1 train and validation annotation archives and downloaded the corresponding COCO photos. These uploads contain JSON annotations, not images; the delivered candidate archive includes actual JPEGs and converted labels.

| Split | Downloaded images | Plate boxes | Images flagged non-exhaustive |
| --- | ---: | ---: | ---: |
| Train | 328 | 957 | 38 |
| Validation | 60 | 170 | 6 |
| Total | 388 | 1,127 | 44 |

Source category ID 756 (`paper_plate`) maps to project class ID 1 (`plate`). Official LVIS splits were preserved, following each image's coco_url; LVIS validation images can come from the COCO train2017 directory. All 1,127 source annotations include segmentation masks, retained in the filtered JSON files.

## Three checks

1. Source and structural integrity: the uploaded archives are byte-identical to the official downloads used for extraction. All 388 JPEGs decode and match source width/height; all 1,127 boxes have positive dimensions and pass source bounds checks. The exported YOLO boxes were independently reconstructed and match the source coordinates within 0.000001 pixel. No source rounding clipping was needed. No download failures remain.
2. Duplicate and split screening: zero exact byte/pixel duplicate groups, zero near-duplicate candidate pairs, and no train/validation image-ID overlap within the subset. Against the existing 1,616-image corpus, zero exact byte/pixel overlap and zero near-match candidates. Near screening used 64-bit dHash distance <=6, aspect ratio difference <=0.05 and 16x16 RGB thumbnail mean absolute error <=15. A screen cannot prove absence of all related scenes.
3. Visual sample: inspected 16 train and 8 validation images evenly spaced by sorted image ID, with source boxes overlaid. They depict disposable plates in varied food, household, restaurant and outdoor settings. This supports general plate relevance, not exact NHS Eco-Products EP-P013 identity. The remaining 364 images have not received visual review.

## Candidate status

The bundle is `LVIS_Paper_Plate_Candidate.zip`, with images, labels, filtered original annotations/masks, per-image provenance/checksums/license IDs, full verification JSON, two visual sample sheets, and lists excluding the 44 plate-non-exhaustive images.

Other cafeteria target classes are not labeled in the YOLO export. Before adding this to the project's accepted 11-class corpus, review every image, repair missing plate labels, label other visible target objects, and confirm the chosen plate-domain policy. The existing accepted corpus remains 1,616 images / 3,824 boxes; its accepted plate count is still zero. No training or NHS performance claim is made. The new plate candidate pool contains 388 images / 1,127 source instances.

Paper boats and exact NHS four-compartment containers remain unfilled dataset gaps. Product pages found during the web search are references, not accepted datasets. LVIS paper_plate does not resolve those classes.

## Sources and rights

Official annotation files:
- https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip
- https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip
- https://github.com/lvis-dataset/lvis-api

SHA-256 of uploaded/source archives:
- Train: `334a4caa374030a7817cf050364525e910f7960f9b6968cef47cffbf3893f8ba`
- Validation: `5cae9a3c79aadb667550c2b5dcf7f4d86e059a41ec91ef690225b667e28e9ba5`

LVIS annotations are CC BY 4.0. Images retain their individual COCO/Flickr terms; the annotation license does not relicense images. Per-image license IDs and source URLs are included. See https://cocodataset.org/#termsofuse.

GitHub records this verification report. Actual candidate photos, labels and metadata are delivered in the downloadable ZIP, outside the accepted corpus.
