# Dataset audits

The current model has **10 classes** (see `config.yaml`). Run `python3 src/audit.py --manifests data/manifests --output data/audits/current` to regenerate the audit after changing the data. The old audit lives at `data/audits/old_11class_snapshot/` and documents the historical 11-class labels before TartanHacks images were included.

**Source-level screens are not proof of complete annotations.** Examples can have unannotated target objects; correlated images/video frames can inflate validation scores. Preserve capture groups and hold out independent natural NHS scenes for final testing.
