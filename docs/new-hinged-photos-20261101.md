# New hinged-container image sourcing — 2026-11-01

## Downloaded outputs (actual GitHub Actions artifacts)

- [Curated **29-photo batch** (21 likely hinged containers; 8 negatives), ~6.9 MB](https://github.com/cx-57/yolo-cafeteria-research/actions/runs/41332845065/artifacts/13093654495).
  Includes `images/`, `manifest.json` with Commons license/creator/source, `index.html`, and `contact-sheet.jpg`.
- [Original **64-photo source pool**, ~15.6 MB](https://github.com/cx-57/yolo-cafeteria-research/actions/runs/41332494158/artifacts/13093033962).
  Includes 18 from foam-food-boxes, 18 from other foam-container sources, 10 from plastic food boxes, 10 from disposable lunch boxes, and eight expanded-polystyrene tray sources.
- These artifacts expire **2026-12-01**. The GitHub Actions workflow can stage a new batch later, but it may not be identical when upstream sources change.

The 64-image pool was downloaded via the Wikimedia Commons API, using per-image rights metadata that advertises CC BY, CC0, or public-domain licenses. The automated collection **excluded the 38 titles from the prior Commons artifact** and checked byte SHA-256 hashes. Source licensing/attribution still require individual verification before reuse/publication. Cross-checking against the original private Mac historical training collection is NOT complete.

The 29-photo selection was based on qualitative review of the new contact sheet (not on their detector scores) and is **provisional**: 21 photographs appear to show open, closed, foam or plastic food boxes/clamshells; eight are candidate hard negatives (trays, food-on-plates, or other similar-looking non-clamshells). They are not human-adjudicated class labels. One photograph can contain multiple instances; original capture groups and near-duplicate scenes have not been independently certified.

### Before any new training

1. Inspect all 21 likely container photos individually to verify they really depict the target **hinged** class. Some closed boxes may obscure the hinge; mark uncertainty rather than assuming.
2. Annotate every visible target clamshell in YOLO box format, and maintain correctly labeled `food` and `drink_carton` objects on these photos as required by the original three-class specialist. On hard-negative photos, don't mark a visible plate as hinged.
3. Check these source images against *all* existing Mac train, validation and test photos for near-duplicate captures—not only SHA matches. Keep these newly reviewed public photographs in a **new training-only partition**, not the locked evaluation cohort.
4. Make a new candidate, keeping the original four specialists frozen; compare against the unchanged validation metrics and use the 12 public hinged-source photos for qualitative diagnostics only.

**No new model training or improved mAP has occurred during this acquisition.**

## New AI-visual draft annotations for a separate training candidate

- [Download the 29-photo annotated YOLO train-only ZIP from GitHub (11.2 MB)](https://github.com/cx-57/yolo-cafeteria-research/actions/runs/41334026397/artifacts/13094591299).
- 29 photos, **24 hinged-container boxes** across **22 photos**, **28 food boxes**, and **7 hard-negative photos** (one entirely background-only). Zero drink cartons appear in the new photos.
- Class IDs in the ZIP are specific to the original *three-class* food/container/carton specialist: `0=food, 1=hinged_container, 2=drink_carton`.
- Image/checksum/YOLO-label validation passed on GitHub Actions; red/green overlays and source attribution are included in the ZIP. The annotations are **AI-assisted draft visual labels** and **not independently human verified**. Some closed boxes and mixed meals need additional expert review. This dataset is for **development training only**, not ground-truth validation or test scoring.

### One-command Mac experiment (v3)

Download the GitHub annotated ZIP above to your Mac's `Downloads` folder, then:

```bash
cd ~/Desktop/yolo-cafeteria-research
git pull --ff-only origin main

./scripts/run_specialists.sh hinge all \
  --out "$HOME/Desktop/yolo-cafeteria-research-assets/artifacts/hinged-candidate-v3-newphotos" \
  --device mps \
  --bundle "$HOME/Downloads/cafeteria-44-image-review-kit-github.zip" \
  --extra-annotated "$HOME/Downloads/hinged-29-annotated-train-only-AI-DRAFT.zip"
```

This creates **new candidate weights**, updates neither the frozen baseline nor original training labels, checks exact-byte overlaps with existing train/validation/test photos and the 44-photo stress bundle, and produces frozen-development validation and 12-photo side-by-side stress comparisons. Historical *near duplicates* remain a limitation; exact hash non-overlap is not an independence certificate.

Inspect:
`.../hinged-candidate-v3-newphotos/validation-gate.json` and
`.../hinged-candidate-v3-newphotos/stress-comparison/comparison.json`.
Neither establishes independent accuracy; hinged-container support in the corrected 186-image validation set is only two boxes.

