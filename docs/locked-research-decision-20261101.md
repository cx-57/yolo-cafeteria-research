# Locked research decision — November 1, 2026

**Decision:** Preserve the existing four frozen YOLO specialists and owner-selected disposal policy. Do **not** retrain, change inference thresholds, modify class mappings, or reinterpret developmental results based on the 44-photo exploratory stress test. Keep the original research freeze in `configs/research-freeze.json` unchanged.

## Recorded exploratory observation
- 44 public-source candidate photos; 149 displayed object detections at the original display floor.
- 12 compost, 49 recycling, and 88 review outputs; these are prediction counts, not correct-bin classifications.
- Three images had no displayed detections.
- Source-category *image-level* class matches: bottle 9/9; drink carton 5/6; utensil 12/15; hinged container 4/12; wrapper 0/2. These are **not recall or accuracy** because source hints are not checked object annotations.
- Qualitative concerns: plate/hinged-container confusion, missing wrappers, overlapping boxes in crowded scenes, and a foil meal tray misdetected as a drink carton at confidence 0.617 with an automatic recycling suggestion.
- There is **no independent mAP or actual school-bin correctness estimate**, and no human-verified ground-truth labels for this cohort. This must remain explicit in any manuscript or summary.

## Next work, separately scoped
- Keep the original model/policy and the paper's existing 248-photo development evaluation distinct from any future experimental retraining.
- If developing a new candidate version, prioritize hinged containers and wrappers, evaluate hard negatives including foil trays, and investigate overlapping predictions; do not present improvements without a verified comparison.
- For independent field claims, reserve new school capture days and verify labels/bin eligibility before reporting performance.

**Sources:** [visual audit](exploratory-44-visual-audit.md), [research paper draft](research-paper.md), [original pinned freeze](../configs/research-freeze.json).

This is a documented research decision, not a technical repository lock or an independent validation certificate.
