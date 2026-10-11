# TartanHacks v2 curated subset

Source: https://universe.roboflow.com/charles-xie/tartanhacks-ftewu/dataset/1
License: CC BY 4.0 (original dataset attribution required)

35 napkin frames remapped source class 3 → project class 6; 27 disposable utensil frames remapped source class 8 → project class 3. Images are a deliberately thinned subset of sequential video frames. Each capture group must stay in one split to avoid train/validation/test leakage. Other classes from the original export are not accepted here. Provisional: labels require complete multi-class review before final training. See `data/manifests/tartanhacks.csv` and `data/audits/source_screening_2026-10-10.md`.
