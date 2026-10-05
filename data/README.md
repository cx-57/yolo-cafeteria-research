# Data layout

Dataset bytes are intentionally not committed.

Expected local structure:

```text
data/
├── manifests/
├── raw/
│   ├── public/
│   └── nhs/
├── interim/
├── processed/
│   ├── images/{train,val,test}/
│   └── labels/{train,val,test}/
└── splits/
```

Rules:
1. Preserve original downloads under `raw/`.
2. Transform only into `interim/` or `processed/`.
3. Track every retained image in `manifests/image_manifest.csv`.
4. Never allow near-duplicate scenes across train/validation/test.
5. Freeze the final NHS test set before model selection.
