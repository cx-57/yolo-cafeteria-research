# V5 evaluation diagnostic and balanced-split follow-up

Updated 2026-11-05. These results were reported from a V5 Colab evaluation, but the raw evaluation outputs/checkpoint have not been independently reproduced in this repository. Do not present them as a final benchmark.

## The problem with the original V5 split

- 2,261 images: **2,093 train (92.6%), 97 validation (4.3%), 71 test (3.1%)**.
- The 97-image validation partition had **no labeled hinged containers, drink cartons or bottles**, and just **13 wrapper instances**. Therefore it could not measure all seven classes.
- 48 labeled utensil instances appeared in 29 validation images, producing poor recall with substantial sampling sensitivity.

## Previously reported four-class validation

| Class | Reported val mAP50 | Interpretation |
|---|---:|---|
| food | 81.29% | Measured on original limited validation split |
| plate | 93.90% | Measured on original limited validation split |
| utensil | **26.21%** | Weak, with reported precision 64.3% and recall **16.7%** (roughly 8 of 48 detected at reported recall operating point) |
| hinged_container | N/A | No labeled validation instances |
| drink_carton | N/A | No labeled validation instances |
| wrapper | 80.06% | Only 13 labeled validation instances |
| bottle | N/A | No labeled validation instances |

### Utensil diagnosis (hypotheses, not proven causes)

The most notable issue is **low recall**, suggesting many missed utensils. Potential explanations to test with prediction overlays and per-instance confidence scores are small/thin object size at image resolution, clutter and partial occlusion, limited appearance/domain diversity, augmentation sensitivity, and any residual cross-class annotation gaps. Much of the original corpus underwent provisional AI/worker annotation review; **do not automatically attribute the low recall to a generally unreviewed dataset**. The [annotation audit](ANNOTATION_AUDIT_STATUS.md) records unresolved worker flags, a small number of withheld ambiguous reclassifications and outstanding V4 cross-class coverage, so label quality is not guaranteed either.

### Corrective experiment

Use **[YOLO_V5_Balanced_80_10_10_20_Epochs.ipynb](YOLO_V5_Balanced_80_10_10_20_Epochs.ipynb)**. It:

- Targets **1,809 train / 226 val / 226 test** (approximately 80/10/10) from the **same 2,261** photos. Exact counts depend on duplicate groups.
- Keeps the **original 97 val and 71 test images outside training**, while adding old-training images into the larger held-out groups.
- Groups known identical / conservative near-duplicate images and manifest pair/tray/meal IDs; fails if it cannot safely construct a roughly 80/10/10 dataset or if any of the seven classes lacks labels in val or test.
- Trains **fresh YOLO11s, 20 epochs, 800px, batch 4** and writes per-class validation metrics, class-support counts and split provenance.
- **Does not retrain or reannotate by itself**, does not guarantee group independence for every camera session, and does not certify cross-class label completeness or independently human-verified test labels.

**Do not directly compare the new validation score to the historical 97-image score as though the validation set were unchanged.** The new balanced-split experiment is exploratory; compare against a baseline retrained on that same fixed new split if evaluating architecture or augmentation changes.
