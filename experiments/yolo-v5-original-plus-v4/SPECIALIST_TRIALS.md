# Explicit V5 specialist trials

The user requested specialists for the weaker classes: utensil, drink carton and wrapper. This experiment starts from the existing trained V5, rather than the weak fresh five-epoch baseline.

| Component | Owned classes |
| --- | --- |
| Unified trained V5 | Food, plate, hinged container, plus any class whose expert fails selection |
| Utensil expert | Utensil (native 0 → global 2) |
| Carton expert | Drink carton (native 0 → global 4) |
| Wrapper expert | Wrapper (native 0 → global 5) |

Local expert initializations and training views have been created. They retain the trained backbone and selected output weights; **no additional specialist training epochs have completed locally**, because disk/swap space blocked the run. They are not advertised as improved models.

Open `YOLO_V5_Specialists_5Epochs.ipynb` in Colab, select a T4 GPU, and upload the original 11-epoch `best.pt` plus `YOLO_V5_partial_annotation_audit320(5).zip`. The notebook checks both SHA256 values, recreates the authorized six-class data, evaluates the trained parent, trains each specialist for five epochs, and compares the full hybrid. Images and label views stay on Colab's local disk; checkpoints and reports can persist in Drive.

All positive training images are retained, with up to one deterministic negative per positive. Each expert uses the unchanged 226-image validation cohort, including target-negative images. Test data is unused. The initialized local views contain 910 utensil, 1809 carton, and 590 wrapper training images; their validation annotations are 96, 118 and 43 respectively.

Accept an expert only for at least +1 percentage point AP50–95 while allowing no more than 2 points AP50 regression. Replace the unified predictions for that class; do not concatenate duplicate ownership. Compute all six classes' AP again from the complete merged predictions. This is a development-model selection rule, not proof of independently measured improvement.

CLI alternative, using a fresh output directory:

```bash
python scripts/train_v5_specialists.py \
  --parent /absolute/path/original-best.pt \
  --dataset /absolute/path/prepared-six-class-data \
  --output /absolute/path/specialist-experiment \
  --epochs 5 --device 0 --batch 4
```

`--prepare-only` creates weights and label views without training. Verified completed evaluations can be reused. If interrupted during training, preserve `last.pt`, `best.pt`, configuration and logs for recovery; the runner refuses to overwrite or silently restart an incomplete training directory.

The original V5 training split is unavailable; prior exposure of the present validation images is unresolved. Audit320 is provisional and lacks the later label corrections. Neither the unified nor hybrid result should be presented as independent field accuracy. Deployment inference also needs its chosen thresholds and latency evaluated; the comparison uses standard multi-label validation NMS.
