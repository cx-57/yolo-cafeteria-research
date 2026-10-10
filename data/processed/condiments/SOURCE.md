# Sauce packets — reviewed condiment_packet source

Source: https://universe.roboflow.com/charles-xie/sauce-a1ypv-d8bii/dataset/1
License: CC BY 4.0
Uploaded export: sauce.v1-v2.yolov11.zip

The data.yaml metadata identifies charles-xie / sauce-a1ypv-d8bii, version 1. The export README title says v1 v2; source identity/version follows data.yaml.

All 44 images were visually inspected in contact sheets. They depict single-serve tomato/chili sauce sachets compatible with the target object form. Keep all 44 as public-source training candidates; exact NHS packet identity is unverified.

- Final class: 8 = condiment_packet
- Source classes collapsed: chili (23 images), tomato (21 images)
- Images/instances: 44 / 44
- Native detection boxes: 43
- Segmentation polygons converted to bounding boxes: 1
- Historical splits: train 34, valid 10
- Exact duplicates: 0
- Near-duplicate candidates within this source: 8 pairs

The images mostly show two packet designs on the same wooden background. Keep this entire batch together when splitting; use independent NHS images for validation and final evaluation. Do not count these views as 44 independent packaging designs or assume the old source split measures deployment generalization.

Audit: data/audits/condiments_2026-10-09/
Manifest: data/manifests/condiments.csv
