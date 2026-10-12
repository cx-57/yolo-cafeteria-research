# Dataset audits

The current model has **11 classes** (see `config.yaml`). Run `python3 scripts/audit.py --manifests data/manifests --output data/audits/current` to regenerate the audit after changing the data. The old audit lives at `data/audits/old_11class_snapshot/` and documents the historical labels before TartanHacks images were included.

The [October 11 performance investigation](model_plateau_2026-10-11/README.md)
explains the 73.8% mAP50 run and provides an annotation review queue, split
leakage evidence, and controlled experiment commands.

**Source-level screens are not proof of complete annotations.** Examples can have unannotated target objects; correlated images/video frames can inflate validation scores. Preserve capture groups and hold out independent natural NHS scenes for final testing.
